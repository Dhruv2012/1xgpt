import json
import math
import os
import random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset as TorchDataset

from genie.factorization_utils import factorize_token_ids, unfactorize_token_ids
from genie.config import GenieConfig
from genie.st_mask_git import cosine_schedule


class RawTokenDataset(TorchDataset):
    """ Loads raw tokens as memmap-backed array (v1 single-file) or sharded (v2.0). """
    def __init__(
        self,
        data_dir,
        window_size,
        stride=1,
        filter_interrupts=True,
        filter_overlaps=False
    ):
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.num_shards = 0
        video_bin_path = data_dir / "video.bin"
        if video_bin_path.exists():
            # ---------------- v1 (authors' original logic) ----------------
            self.is_sharded = False

            shape = (self.metadata["num_images"], self.metadata["s"], self.metadata["s"])
            video_tokens_path, segment_ids_path, action_tokens_path = [data_dir / f"{name}.bin"
                                                                       for name in ["video", "segment_ids", "actions"]]
            token_dtype = np.dtype(self.metadata.get("token_dtype", "uint32"))
            self.data = np.memmap(video_tokens_path, dtype=token_dtype, mode="r", shape=shape)

            if os.path.isfile(segment_ids_path):
                self.segment_ids = np.memmap(
                    segment_ids_path, dtype=np.int32, mode="r", shape=(self.metadata["num_images"],)
                )
            else:
                self.segment_ids = None
                if filter_interrupts:
                    raise NotImplementedError("Cannot filter interrupted sequences without segment ids.")

            s = int(self.metadata["s"])
            self.S = s * s

        else:
            # ---------------- v2 (sharded; Cosmos DV8x8x8) ----------------
            # Layout:
            #   - Each shard: videos/video_{shard}.bin contains DV clip tokens with shape
            #       (num_clips, 3, 32, 32), dtype=int32
            #   - One clip = 17 frames @ 30Hz (fixed DV window)
            #   - Segment ids remain frame-level: segment_idx_{shard}.bin (int32, shape=(num_frames,))
            self.is_sharded = True
            self.data_dir = data_dir
            self.videos_dir = data_dir / "videos"
            self.segment_indices_dir = data_dir / "segment_indices"

            # Cosmos DV constants
            self.frames_per_clip = 17
            self.spatial_side = 32
            self.num_groups = 3
            self.factored_vocab_size = 65536
            self.S = self.spatial_side * self.spatial_side
            self.token_dtype = np.dtype("int32")

            # Load per-shard metadata and compute clip counts
            self.shard_metadata: List[dict] = []
            self.shard_cumulative_clips: List[int] = [0]
            metadata_dir = data_dir / "metadata"
            shard_idx = 0
            while (metadata_dir / f"metadata_{shard_idx}.json").exists():
                with open(metadata_dir / f"metadata_{shard_idx}.json") as f:
                    m = json.load(f)
                if "shard_num_frames" not in m:
                    raise ValueError(f"metadata_{shard_idx}.json missing 'shard_num_frames'")
                frames = int(m["shard_num_frames"])
                clips = (frames + self.frames_per_clip - 1) // self.frames_per_clip
                m["_num_clips"] = clips
                self.shard_metadata.append(m)
                self.shard_cumulative_clips.append(self.shard_cumulative_clips[-1] + clips)
                shard_idx += 1
            self.num_shards = len(self.shard_metadata)
            if self.num_shards == 0:
                raise ValueError(f"v2: no shard metadata found in {metadata_dir}")
            self.total_clips = self.shard_cumulative_clips[-1]

            # Holders
            self.current_shard_idx = -1
            self.current_shard_data = None  # (clips, 3, 32, 32)
            self.current_shard_segment_ids = None

        # ---------------- common init ----------------
        self.window_size, self.stride = window_size, stride  # in CLIPS for v2
        self.video_len = (self.window_size - 1) * self.stride  # measured in clips

        self.valid_start_inds = []
        if getattr(self, "is_sharded", False):
            # Build start indices over clips
            total = self.total_clips
            for start_clip in range(0, total - self.video_len):
                if not filter_interrupts or not self.segment_indices_dir.exists():
                    self.valid_start_inds.append(start_clip)
                    continue
                # conservative: require the first-frame segment id to match across window
                s0 = self._clip_segment_id(start_clip)
                s1 = self._clip_segment_id(start_clip + self.video_len)
                if s0 == s1:
                    self.valid_start_inds.append(start_clip)
        else:
            # Build start indices (v1) with original semantics
            for start_ind in range(len(self.data) - self.video_len):
                if not (filter_interrupts and self.segment_ids[start_ind] != self.segment_ids[start_ind + self.video_len]):
                    self.valid_start_inds.append(start_ind)

            if filter_overlaps:
                # Ensure each frame appears at most once (original greedy subset)
                filtered_start_inds = []
                for start_ind in self.valid_start_inds:
                    overlapping_start_inds = {start_ind - i * self.stride for i in range(1, self.window_size)}
                    # Exclude if any overlapping start has already been used
                    for existing_start_ind in filtered_start_inds[-self.window_size * self.stride:]:
                        if existing_start_ind in overlapping_start_inds:
                            break
                    else:
                        filtered_start_inds.append(start_ind)
                self.valid_start_inds = filtered_start_inds

    def _get_shard_idx(self, clip_idx: int) -> int:
        for i in range(len(self.shard_cumulative_clips) - 1):
            if self.shard_cumulative_clips[i] <= clip_idx < self.shard_cumulative_clips[i + 1]:
                return i
        return len(self.shard_cumulative_clips) - 2

    def _load_shard_data(self, shard_idx: int):
        if not getattr(self, "is_sharded", False):
            return
        if self.current_shard_idx == shard_idx:
            return
        # Map DV clips: (num_clips, 3, 32, 32), dtype=int32
        video_path = self.videos_dir / f"video_{shard_idx}.bin"
        num_clips = self.shard_metadata[shard_idx]["_num_clips"]
        self.current_shard_data = np.memmap(
            video_path, dtype=np.int32, mode="r", shape=(num_clips, self.num_groups, self.spatial_side, self.spatial_side)
        )
        # Segment ids: per-frame (optional)
        if self.segment_indices_dir.exists():
            seg_path = self.segment_indices_dir / f"segment_idx_{shard_idx}.bin"
            if seg_path.exists():
                frames = int(self.shard_metadata[shard_idx]["shard_num_frames"])
                self.current_shard_segment_ids = np.memmap(seg_path, dtype=np.int32, mode="r", shape=(frames,))
            else:
                self.current_shard_segment_ids = None
        else:
            self.current_shard_segment_ids = None
        self.current_shard_idx = shard_idx

    def _clip_segment_id(self, global_clip_idx: int) -> int:
        """Return segment id of the *first frame* of the clip (for filtering)."""
        if not self.segment_indices_dir.exists():
            return 0
        shard_idx = self._get_shard_idx(global_clip_idx)
        self._load_shard_data(shard_idx)
        in_shard_clip = global_clip_idx - self.shard_cumulative_clips[shard_idx]
        start_frame = in_shard_clip * self.frames_per_clip
        if self.current_shard_segment_ids is None:
            return 0
        return int(self.current_shard_segment_ids[min(start_frame, len(self.current_shard_segment_ids)-1)])

    def _read_clip(self, global_clip_idx: int) -> np.ndarray:
        """Read fused token grid for a single DV clip -> (32, 32) int64."""
        # breakpoint()
        shard_idx = self._get_shard_idx(global_clip_idx)
        self._load_shard_data(shard_idx)
        in_shard_clip = global_clip_idx - self.shard_cumulative_clips[shard_idx]
        q = self.current_shard_data[in_shard_clip]  # (3, 32, 32)
        V = self.factored_vocab_size  # FIX: use configured base instead of hard-coded 64000
        # fuse q0,q1,q2 into a single integer per (h,w)
        # fused = (q[0].astype(np.int64) + q[1].astype(np.int64) * V + q[2].astype(np.int64) * (V * V))
        # return fused  # (32, 32)
        return (q.astype(np.int64)-1)

    # -------- Torch Dataset API --------
    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        start_ind = self.valid_start_inds[idx]
        print("start_ind", start_ind, "len(self.valid_start_inds)", len(self.valid_start_inds))
        if getattr(self, "is_sharded", False):
            T = self.window_size
            clips = [self._read_clip(start_ind + k * self.stride) for k in range(T)]  # list of (32,32)
            x = torch.from_numpy(np.stack(clips, axis=0).astype(np.int64))  # (T*3, 32, 32)
        else:
            x = torch.from_numpy(
                (self.data[start_ind : start_ind + self.video_len + 1 : self.stride]).astype(np.int64)
            )  # (T, s, s)
        x = x.flatten()  # (T*S,), S=1024 for v2
        attention_mask = torch.ones_like(x)
        return {
            "input_ids": x,
            "labels": x,
            "attention_mask": attention_mask,
        }


def get_maskgit_collator(config: GenieConfig):
    mask_token_id = config.image_vocab_size
    h = w = math.isqrt(config.S)

    def collate_fn(features) -> dict[str, torch.Tensor]:
        # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
        # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1

        input_ids = torch.stack([ex["input_ids"] for ex in features])
        device = input_ids.device
        print("t ", config.T, "h ", h, "w", w, "b ", len(features))
        x_THW = rearrange(input_ids, "b (t h w) -> b t h w", b=len(features), t=config.T,
                          h=h, w=w)
        x_THWC = factorize_token_ids(x_THW, config.num_factored_vocabs, config.factored_vocab_size)
        labels = x_THW.clone()

        # As done in Copilot-4D paper, add random noise sampled with a random rate between 0% and `config.max_corrupt_rate`
        r = torch.rand(x_THWC.size(), device=device)
        u01 = torch.rand((), device=device)
        random_patches_mask = r < config.max_corrupt_rate * u01
        random_values = torch.randint(low=0, high=config.factored_vocab_size, size=x_THWC.size(),
                                      dtype=torch.long, device=device)
        x_THWC[random_patches_mask] = random_values[random_patches_mask]

        if random.random() < config.non_mlm_ratio:  # Closer to autoregressive inference
            # Leave frames [0, first_masked_frame) unmasked.
            first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
            x_THWC_view = x_THWC[:, first_masked_frame:]

            # Arbitrary numbers here, but corrupting later frames more
            # since we likely have compounding errors.
            correct_rate = random.uniform(0.25, 1.0)
            for i in range(x_THWC_view.size(1)):
                correct_rate *= random.uniform(0.9, 1.0)
                r = torch.rand((len(features), h, w, config.num_factored_vocabs), device=device)
                random_patches_mask = r > correct_rate
                x_THWC_view[:, i][random_patches_mask] = random_values[:, first_masked_frame + i][random_patches_mask]
        else:  # Typical MLM masking
            first_masked_frame = 1

        mask = torch.zeros(1)
        c = 0
        while mask.max() == 0:  # We could get unlucky and mask no tokens?
            # per-minibatch, per-frame masking probability (could try variable masking rate from MUSE)
            mask_prob_T = cosine_schedule(torch.rand(len(features), config.T - first_masked_frame, 1, 1))

            r = torch.rand_like(x_THW[:, first_masked_frame:], dtype=torch.float)
            mask = r < mask_prob_T
            c += 1

        if c > 1:
            print(f"Generated mask {c} > 1 times.")

        x_THW = unfactorize_token_ids(x_THWC, config.num_factored_vocabs, config.factored_vocab_size)
        x_THW[:, first_masked_frame:][mask] = mask_token_id

        return {
            "input_ids": rearrange(x_THW, "b t h w -> b (t h w)"),
            "labels": rearrange(labels, "b t h w -> b (t h w)"),
        }

    return collate_fn
