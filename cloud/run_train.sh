# Change permissions for the script to be executed
chmod +x ../model_training.py

# Make the command immune to hangups ( Like logging out )
nohup python3 ../model_training.py &
