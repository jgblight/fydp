#!/bin/bash
# Fakenect .sh 
#

wordlist=(
	'BIG'
	'CAT'
	'FAVOURITE'
	'HOUSE'
	'MOTHER'
	'MORE'
	'MOVIE'
	'RED'
	'SMALL'
	'SWEETHEART'
       )

clear
echo "Recording data for $1"

for i in "${wordlist[@]}" 
do	
	read -n 1 -s -p "Record $i for $1, press [ENTER]. Press [CTRL+C] then [ENTER] to stop."
	record ~/Dropbox/fakenect-storage/sign_library/$i/$1
	read
	
	echo "Save [y] or try again [n]?"
	read save
	echo
	while [ $save == "n" ] #loop until a good sample is taken
	do
		rm -rf ~/Dropbox/fakenect-storage/sign_library/$i/$1
		read -n 1 -s -p "Record $i for $1, press [ENTER]. Press [CTRL+C] then [ENTER] to stop."
		record ~/Dropbox/fakenect-storage/sign_library/$i/$1
		
		echo "Save [y] or try again [n]?"
		read save
		echo
	done
	cp $2 ~/Dropbox/fakenect-storage/sign_library/$i/$1
done


echo "Complete."
