#!/bin/bash
# Fakenect .sh 
#

clear
echo "recording GARBAGE data for $1"

for i in {1..10}
do
	read -n 1 -s -p "Record GARBABE # $i for $1, press [ENTER]. Press [CTRL+C] then [ENTER] to stop."
	sleep 2
	sudo record ~/Dropbox/fakenect-storage/sign_library/GARBAGE/$1$i
	read
	sudo chown -R sara ~/Dropbox/fakenect-storage/sign_library/GARBAGE/$1$i
	cp $2 ~/Dropbox/fakenect-storage/sign_library/GARBAGE/$1$i
done

echo "Complete."
