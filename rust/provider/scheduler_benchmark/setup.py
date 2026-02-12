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
        "ros2": ["rclpy", "std_msgs"],
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
