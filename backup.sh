#!/bin/bash

# Have to run it after sshing into Garfield!
# DO NOT USE ~ USE $HOME INSTEAD!

srcPath="$HOME/Documents/Thesis/src";

source "$srcPath/thesis/bin/activate";
pip freeze > "$srcPath/requirements.txt";
deactivate;
rsync -avP --exclude-from="$srcPath/.backupignore" "$srcPath/" abahari@login1.cair.mun.ca:~/thesis;

if [ $? -eq 0 ]; then
    echo "$(date +%Y/%m/%d-%H:%M:%S): + Successful Backup!" >> "$srcPath/backup.log";
else
    echo "$(date +%Y/%m/%d-%H:%M:%S): - Backup Failed!" >> "$srcPath/backup.log";
fi