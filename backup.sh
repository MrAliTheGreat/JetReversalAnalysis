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
#
################################################################################ Crontab ################################################################################
#
# To run the backup periodically put the following command in "crontab -e" file --> It'll run the backup.sh at 23:00 every Friday!
#   0 23 * * 5 ~/path-to-backup.sh
#
# Since we are using ssh-agent we need to specify SSH_AUTH_SOCK in crontab file so that it'll know where to look to find the private_key added by ssh-add
# Run the following command and add the output to top of the crontab file:
#   env | grep SSH_AUTH_SOCK
# It'll look something like:
#   SSH_AUTH_SOCK=path-to-directory
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
