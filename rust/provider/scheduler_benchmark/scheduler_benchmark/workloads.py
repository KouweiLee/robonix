# SPDX-License-Identifier: MulanPSL-2.0
"""
Synthetic workload generators for CPU and GPU scheduling benchmarks.

Provides reproducible, configurable workloads that simulate typical embodied AI
computation patterns: path planning, SLAM scan matching, neural network inference,
image processing, audio processing, etc.

Each workload class pre-allocates memory and provides a `run()` method that
performs one iteration of work, returning wall-clock elapsed time in seconds.
"""

import time
import numpy as np
import torch


def gpu_available() -> bool:
    """Check if GPU (CUDA) workloads are supported."""
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# CPU Workloads
# ---------------------------------------------------------------------------

class CPUGemm:
    """
    General matrix multiply (GEMM) on CPU.
    Simulates linear algebra heavy computations found in SLAM, IK solvers,
    and state estimation (EKF/UKF).
    """

    def __init__(self, size: int = 512):
        self.size = size
        # Pre-allocate to avoid measuring allocation overhead
        self._a = np.random.randn(size, size).astype(np.float32)
        self._b = np.random.randn(size, size).astype(np.float32)

    def run(self) -> float:
        """Run one GEMM iteration. Returns elapsed seconds."""
        start = time.perf_counter()
        np.dot(self._a, self._b)
        return time.perf_counter() - start


class CPUPathPlan:
    """
    Simulates costmap-based path planning (like Nav2's NavFn / Dijkstra).
    Performs a wavefront expansion on an occupancy grid, which is the
    dominant computation in Nav2 global planner.
    """

    def __init__(self, grid_size: int = 300):
        self.grid_size = grid_size
        # Create a deterministic occupancy grid (1.0 = free, 0.0 = obstacle)
        np.random.seed(42)  # Fixed seed for reproducibility
        self._grid = np.random.random((grid_size, grid_size)).astype(np.float32)
        self._grid[self._grid < 0.25] = 0.0  # obstacles (~25%)
        self._grid[self._grid != 0.0] = 1.0
        
        # Pre-allocate output buffer to avoid allocation overhead in loop
        # distance_transform_edt requires float64 output
        self._output = np.zeros_like(self._grid, dtype=np.float64)

    def run(self) -> float:
        """
        Run one planning iteration via distance transform.
        Simulates costmap inflation/update which is O(N^2) memory bound.
        """
        import scipy.ndimage
        start = time.perf_counter()
        
        # Simulates a costmap update: compute distance to nearest obstacle
        # varied slightly to prevent perfect caching if OS is smart, 
        # though in this case we want deterministic load.
        # We invert grid because edt computes distance to zero.
        # Use distances parameter for output buffer instead of 'output'
        scipy.ndimage.distance_transform_edt(self._grid, return_distances=True, return_indices=False, distances=self._output)
        
        return time.perf_counter() - start


class CPUScanMatch:
    """
    Simulates LiDAR scan matching for SLAM using cross-correlation in
    frequency domain. This is the core computation in scan-matching SLAM
    approaches (e.g., Cartographer correlative scan matcher).
    """

    def __init__(self, scan_points: int = 360, grid_size: int = 256):
        self.grid_size = grid_size
        np.random.seed(42)
        # Pre-generate multiple frames to simulate robot motion
        # This prevents perfect branch prediction/caching while keeping deterministic load
        self._refs = [np.random.randn(grid_size, grid_size).astype(np.float32) for _ in range(5)]
        self._scans = [np.random.randn(grid_size, grid_size).astype(np.float32) for _ in range(5)]
        self._idx = 0

    def run(self) -> float:
        """
        Run one scan matching iteration via FFT cross-correlation.
        Simulates a Correlative Scan Matcher that searches over multiple orientations.
        This increases the computational density to match real SLAM front-ends.
        """
        start = time.perf_counter()
        
        # Cycle through pre-generated frames
        ref = self._refs[self._idx]
        scan = self._scans[self._idx]
        self._idx = (self._idx + 1) % 5
        
        # Simulate searching over 5 different orientations (rotations)
        # Real CSM would rotate the scan and re-fft, or use log-polar transform.
        # Here we simulate the compute load by running FFT multiple times.
        best_score = -1.0
        for _ in range(5):
            # In a real algo, we'd rotate 'scan' here.
            # For benchmark load, we just repeat the heavy FFT ops.
            ref_fft = np.fft.fft2(ref)
            scan_fft = np.fft.fft2(scan)
            corr = np.fft.ifft2(ref_fft * np.conj(scan_fft))
            score = np.max(np.abs(corr))
            if score > best_score:
                best_score = score
        
        return time.perf_counter() - start


class CPUPointCloud:
    """
    Simulates point cloud processing (downsampling, normal estimation).
    Common in 3D perception pipelines and depth camera processing.
    """

    def __init__(self, num_points: int = 100000):
        self.num_points = num_points
        np.random.seed(42)
        # Pre-allocate point cloud buffers (simulate circular buffer of sensor data)
        self._clouds = [np.random.randn(num_points, 3).astype(np.float32) for _ in range(3)]
        self._idx = 0

    def run(self) -> float:
        """Run point cloud processing: voxel grid filter + PCA normals."""
        start = time.perf_counter()
        
        cloud = self._clouds[self._idx]
        self._idx = (self._idx + 1) % 3
        
        # Voxel grid downsampling simulation
        voxel_size = 0.05
        quantized = np.floor(cloud / voxel_size).astype(np.int32)
        
        # Use structured array for unique voxels - heavy CPU sorting task
        _, idx = np.unique(
            quantized.view(np.dtype((np.void, quantized.dtype.itemsize * 3))),
            return_index=True
        )
        downsampled = cloud[idx]
        
        # Compute covariance for normal estimation (batch)
        # This is heavy linear algebra on small matrices
        n = min(len(downsampled), 5000)
        subset = downsampled[:n]
        centered = subset - subset.mean(axis=0)
        cov = centered.T @ centered / n
        _ = np.linalg.eigh(cov)
        
        return time.perf_counter() - start


class CPUImageProcess:
    """
    Simulates camera image processing pipeline:
    format conversion, resizing, and rectification (undistortion).
    Focuses on high memory bandwidth and pixel-wise remapping.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        np.random.seed(42)
        # Pre-allocate frames
        self._images = [
            np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        self._idx = 0
        
        # Pre-compute rectification maps (simulating camera lens calibration)
        # map_x and map_y describe where each output pixel comes from in the input.
        rows, cols = height, width
        y, x = np.indices((rows, cols), dtype=np.float32)
        
        # Create a simple radial distortion-like map
        cx, cy = width / 2.0, height / 2.0
        dx = (x - cx) / cx
        dy = (y - cy) / cy
        r = np.sqrt(dx**2 + dy**2)
        distortion = 1.0 + 0.2 * r**2  # k1=0.2
        
        self._map_x = cx + dx * distortion * cx
        self._map_y = cy + dy * distortion * cy
        
        # Clip maps to image boundaries
        self._map_x = np.clip(self._map_x, 0, width - 1)
        self._map_y = np.clip(self._map_y, 0, height - 1)

    def run(self) -> float:
        """Run driver-level image pipeline: conversion, resize, rectification."""
        import scipy.ndimage
        start = time.perf_counter()
        
        # 1. Simulate Debayer/Format Conversion (e.g. YUV -> RGB)
        # Heavy memory read/write with simple math
        img_uint8 = self._images[self._idx]
        self._idx = (self._idx + 1) % 3
        
        img = img_uint8.astype(np.float32)
        # Simple weighted sum to simulate color space math
        _ = 0.5 * img + 128.0
        
        # 2. Rectification (Lens Undistortion)
        # This is the most realistic proxy for driver-level CPU load:
        # non-linear memory access via mapping.
        # We do it channel by channel to simulate typical implementation.
        rectified = np.zeros_like(img)
        for c in range(3):
            scipy.ndimage.map_coordinates(
                img[:, :, c], 
                [self._map_y, self._map_x], 
                order=1, 
                mode='nearest',
                output=rectified[:, :, c]
            )
        
        # 3. Downsampling (Simulate thumbnail or pyramid generation)
        # Slicing is zero-copy in numpy, so we do a tiny bit of math to force a copy
        _ = rectified[::2, ::2, :].copy()
        
        return time.perf_counter() - start


class CPUAudioProcess:
    """
    Simulates audio feature extraction (mel spectrogram) for speech processing.
    """

    def __init__(self, sample_rate: int = 16000, duration_ms: int = 500):
        self.n_samples = sample_rate * duration_ms // 1000
        self._audio = np.random.randn(self.n_samples).astype(np.float32)
        # Pre-compute mel filterbank
        n_fft = 512
        n_mels = 80
        self._window = np.hanning(n_fft).astype(np.float32)
        self._n_fft = n_fft
        self._n_mels = n_mels
        # Simple mel filterbank
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        mel_lo = 2595 * np.log10(1 + freqs[0] / 700)
        mel_hi = 2595 * np.log10(1 + freqs[-1] / 700)
        mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
        hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
        self._filterbank = np.zeros((n_mels, len(freqs)), dtype=np.float32)
        for i in range(n_mels):
            lo, center, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
            for j, f in enumerate(freqs):
                if lo <= f <= center:
                    self._filterbank[i, j] = (f - lo) / max(center - lo, 1e-8)
                elif center < f <= hi:
                    self._filterbank[i, j] = (hi - f) / max(hi - center, 1e-8)

    def run(self) -> float:
        """Compute mel spectrogram from audio buffer."""
        start = time.perf_counter()
        hop = self._n_fft // 2
        n_frames = (self.n_samples - self._n_fft) // hop + 1
        # STFT
        frames = np.lib.stride_tricks.as_strided(
            self._audio,
            shape=(n_frames, self._n_fft),
            strides=(self._audio.strides[0] * hop, self._audio.strides[0])
        ).copy()
        frames *= self._window
        spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        # Apply mel filterbank
        mel_spec = spec @ self._filterbank.T
        _ = np.log(mel_spec + 1e-9)
        return time.perf_counter() - start


# ---------------------------------------------------------------------------
# GPU Workloads (PyTorch CUDA required)
# ---------------------------------------------------------------------------

class GPUGemm:
    """
    GPU matrix multiply - simulates dense linear algebra in neural networks.
    Requires CUDA.
    """

    def __init__(self, size: int = 2048):
        self.size = size
        self._a = torch.randn(size, size, device='cuda', dtype=torch.float32)
        self._b = torch.randn(size, size, device='cuda', dtype=torch.float32)

    def run(self) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.mm(self._a, self._b)
        torch.cuda.synchronize()
        return time.perf_counter() - start


class GPUConv2d:
    """
    GPU 2D convolution - simulates CNN perception (object detection, segmentation).
    Requires CUDA.
    """

    def __init__(self, batch: int = 4, in_ch: int = 64, out_ch: int = 128,
                 size: int = 112, kernel: int = 3):
        self.batch = batch
        self._x = torch.randn(batch, in_ch, size, size, device='cuda')
        self._w = torch.randn(out_ch, in_ch, kernel, kernel, device='cuda')
        self._pad = kernel // 2

    def run(self) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.nn.functional.conv2d(self._x, self._w, padding=self._pad)
        torch.cuda.synchronize()
        return time.perf_counter() - start


class GPUTransformerBlock:
    """
    Simulates a transformer inference pass (self-attention + FFN).
    Models workloads like VLA (Vision-Language-Action), speech recognition,
    and vision transformers. Requires CUDA.
    """

    def __init__(self, layers: int = 6, hidden: int = 768,
                 seq_len: int = 128, batch: int = 1):
        self.layers = layers
        self.hidden = hidden
        self.seq_len = seq_len
        self._x = torch.randn(batch, seq_len, hidden, device='cuda')
        # Pre-allocate weight matrices for each layer
        self._qkv_w = [
            torch.randn(hidden, 3 * hidden, device='cuda') / (hidden ** 0.5)
            for _ in range(layers)
        ]
        self._ffn_w1 = [
            torch.randn(hidden, hidden * 4, device='cuda') / (hidden ** 0.5)
            for _ in range(layers)
        ]
        self._ffn_w2 = [
            torch.randn(hidden * 4, hidden, device='cuda') / ((hidden * 4) ** 0.5)
            for _ in range(layers)
        ]

    def run(self) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        x = self._x
        B, S, H = x.shape
        for i in range(self.layers):
            # QKV projection
            qkv = torch.mm(x.view(-1, H), self._qkv_w[i])
            q, k, v = qkv.split(H, dim=-1)
            q = q.view(B, S, H)
            k = k.view(B, S, H)
            v = v.view(B, S, H)
            # Scaled dot-product attention
            attn = torch.bmm(q, k.transpose(1, 2)) / (H ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            out = torch.bmm(attn, v)
            # FFN
            h = torch.mm(out.view(-1, H), self._ffn_w1[i])
            h = torch.relu(h)
            h = torch.mm(h, self._ffn_w2[i])
            x = h.view(B, S, H) + x  # residual
        torch.cuda.synchronize()
        return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Composite Workload Profiles
# ---------------------------------------------------------------------------

class NavigationWorkload:
    """
    Composite workload simulating Nav2 navigation stack:
    path planning + costmap update + controller loop.
    """

    def __init__(self, grid_size: int = 300, scan_grid: int = 200):
        self.planner = CPUPathPlan(grid_size)
        self.scan_match = CPUScanMatch(grid_size=scan_grid)

    def run(self) -> float:
        """One navigation iteration: plan + localization update."""
        t1 = self.planner.run()
        t2 = self.scan_match.run()
        return t1 + t2


class VLAWorkload:
    """
    Composite workload simulating Vision-Language-Action model inference:
    image preprocessing + transformer inference + action decoding.
    """

    def __init__(self, image_size: int = 224, layers: int = 12,
                 hidden: int = 768, seq_len: int = 256):
        self.image_proc = CPUImageProcess(image_size, image_size)
        self.transformer = GPUTransformerBlock(layers, hidden, seq_len)
        self.action_gemm = CPUGemm(128)  # action space decoding

    def run(self) -> float:
        """One VLA iteration: preprocess + infer + decode."""
        t1 = self.image_proc.run()
        t2 = self.transformer.run()
        t3 = self.action_gemm.run()
        return t1 + t2 + t3


class HeavyPerceptionWorkload:
    """
    Composite workload simulating full perception (Foreground Skill):
    Camera prep + CNN detection + PointCloud analysis.
    Used by 'bench_inspect' skill.
    """

    def __init__(self):
        self.image_proc = CPUImageProcess(640, 480)
        self.detector = GPUConv2d(batch=2, in_ch=64, out_ch=128, size=112)
        self.pointcloud = CPUPointCloud(50000)

    def run(self) -> float:
        t1 = self.image_proc.run()
        t2 = self.detector.run()
        t3 = self.pointcloud.run()
        return t1 + t2 + t3


class CameraStreamWorkload:
    """
    Composite workload simulating camera driver + preprocessing pipeline (Background Process):
    Debayering, resizing, rectification.
    Pure CPU workload, high memory bandwidth, continuous.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.image_proc = CPUImageProcess(width, height)

    def run(self) -> float:
        return self.image_proc.run()


class LidarStreamWorkload:
    """
    Composite workload simulating LiDAR driver + preprocessing pipeline (Background Process):
    Packet decoding, coordinate transformation (TF), and point cloud assembly.
    Focuses on intensive byte manipulation and small matrix-vector multiplications.
    """

    def __init__(self, num_points: int = 30000):
        self.num_points = num_points
        # Pre-allocate transform matrices for 32-line LiDAR-like scan
        self._tf_matrix = np.eye(4, dtype=np.float32)
        # Simulated raw UDP packets: num_points * 32 bytes (distance, intensity, azimuth, etc.)
        self._raw_data = np.random.bytes(num_points * 32) 
        self.point_proc = CPUPointCloud(num_points)

    def run(self) -> float:
        """Run LiDAR driver pipeline: packet decoding, TF, and point processing."""
        start = time.perf_counter()
        
        # 1. Simulate Packet Decoding
        # Iterate over "packets" and extract values (simulated by heavy slicing/viewing)
        # This represents the byte manipulation needed to extract X,Y,Z from raw wire format.
        _ = np.frombuffer(self._raw_data, dtype=np.uint8).reshape(-1, 32) # type: ignore
        
        # 2. Coordinate Transformation (TF)
        # Apply [R|t] to points. This is O(N) matrix-vector math.
        # We simulate this by doing a batch dot product.
        points = np.random.randn(self.num_points, 3).astype(np.float32)
        _ = points @ self._tf_matrix[:3, :3].T + self._tf_matrix[:3, 3]
        
        # 3. Downsampling & Normals (The original workload)
        _ = self.point_proc.run()
        
        return time.perf_counter() - start


class SLAMWorkload:
    """
    Composite workload simulating SLAM:
    scan matching + pose graph optimization (simulated via matrix ops).
    """

    def __init__(self):
        self.scan_match = CPUScanMatch(grid_size=256)
        self.graph_opt = CPUGemm(256)  # Simulates Gauss-Newton solve

    def run(self) -> float:
        t1 = self.scan_match.run()
        t2 = self.graph_opt.run()
        return t1 + t2


class LidarSLAMWorkload:
    """
    Composite workload simulating a tightly-coupled LiDAR-SLAM pipeline:
    LiDAR driver (decoding + TF) + SLAM (scan matching + pose graph).
    Used as a merged background process.
    """

    def __init__(self, num_points: int = 30000):
        self.lidar = LidarStreamWorkload(num_points)
        self.slam = SLAMWorkload()

    def run(self) -> float:
        t1 = self.lidar.run()
        t2 = self.slam.run()
        return t1 + t2


class SpeechWorkload:
    """
    Composite workload simulating speech recognition:
    audio feature extraction + transformer decoder inference.
    """

    def __init__(self):
        self.mel = CPUAudioProcess(16000, 500)
        self.decoder = GPUTransformerBlock(layers=4, hidden=512, seq_len=64)

    def run(self) -> float:
        t1 = self.mel.run()
        t2 = self.decoder.run()
        return t1 + t2


class MotionPlanWorkload:
    """
    Composite workload simulating arm motion planning:
    collision checking + trajectory optimization.
    """

    def __init__(self):
        self.collision = CPUGemm(256)  # Distance queries in configuration space
        self.trajectory = CPUGemm(384)  # Trajectory optimization

    def run(self) -> float:
        t1 = self.collision.run()
        t2 = self.trajectory.run()
        return t1 + t2
