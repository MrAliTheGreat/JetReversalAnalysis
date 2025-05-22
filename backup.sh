#!/bin/bash

# DO NOT USE ~ USE $HOME INSTEAD!

############################################################################# Prerequisites #############################################################################
#
# Destination Server Side:
#   chmod go-w ~
#   chmod 700 ~/.ssh
#   chmod 600 ~/.ssh/authorized_keys
#
# Client Side:
#   ssh-keygen -t ed25519 -a 100
#   cat ~/.ssh/public_key | ssh username@destination_server_ip "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys" --> Make sure authorized_keys has 600 permission!
#   eval "$(ssh-agent -s)"
#   ssh-add ~/.ssh/private_key
#   ssh -fN -L local_port:destination_server_ip:22 username@middle_server_ip --> Create a tunnel from local to destination through middle server
#
# Everything ready! Just run: ssh -p local_port username@destination_server_ip
#
#########################################################################################################################################################################

srcPath="$HOME/Documents/Thesis/src";

source "$srcPath/thesis/bin/activate";
pip freeze > "$srcPath/requirements.txt";
deactivate;
rsync -avP --exclude-from="$srcPath/.backupignore" "$srcPath/" -e 'ssh -p 8569' localhost:~/thesis;

if [ $? -eq 0 ]; then
    echo "$(date +%Y/%m/%d-%H:%M:%S): + Successful Backup!" >> "$srcPath/backup.log";
else
    echo "$(date +%Y/%m/%d-%H:%M:%S): - Backup Failed!" >> "$srcPath/backup.log";
fi