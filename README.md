# Carla_Script
Script for extracting semantic segmentation/depth prediction dataset out of Carla Urban Driving Simulator.

## Command

```bash
# from the folder /carla/CARLA_8_4/PythonClient
# Run the followin in d/nt terminals
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 CarlaUE4.sh
python3.6 client_example_matrix.py --dataset-path=/habtegebrialdata/Datasets/carla
```
