# Upload dataset on ClearML

clearml-data create --project kornaeva-rnf/GA_PINN_3D --name trained_models --storage s3://api.blackhole2.ai.innopolis.university:443/kornaeva-rnf

clearml-data add --files trained_models

clearml-data close

# Run experiments on ClearML

clearml-task --project kornaeva-rnf/GA_PINN_3D --name ga_pinn_3d --script main.py --queue e0841e72c8a544efa9c54b5e768b1683 --docker "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel" --requirements "req.txt" --folder '' --skip-task-init --docker_bash_setup_script setup.sh

# Sync dataset

clearml-data sync --project kornaeva-rnf/GA_PINN_3D --name trained_models --folder trained_models --storage s3://api.blackhole2.ai.innopolis.university:443/kornaeva-rnf