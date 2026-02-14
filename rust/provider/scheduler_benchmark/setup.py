from setuptools import setup, find_packages

setup(
    name="scheduler_benchmark",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
    ],
    extras_require={
        "gpu": ["torch"],
        # rclpy, std_msgs, and robonix_sdk are provided by the sourced
        # ROS 2 workspace and cannot be pip-installed.
    },
    entry_points={
        "console_scripts": [
            "bench-runner=scheduler_benchmark.runner:main",
            "bench-report=scheduler_benchmark.report:main",
            "bench-bg=scheduler_benchmark.background:main",
        ],
    },
    python_requires=">=3.8",
)
