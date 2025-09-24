
run_name="$1"
step="$2"
rsync BeijingH800:~/TG-Interpolation/saved_models/$run_name ~/TG-Interpolation/saved_models/ --exclude wandb --exclude *-unsharded -avzP
rsync BeijingH800:~/TG-Interpolation/saved_models/$run_name/$step ~/TG-Interpolation/saved_models/$run_name --exclude wandb -avzP
