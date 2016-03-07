#!/bin/bash

if [[ -z "$1" || -z "$2" || -z "$3" ]];
 then
        echo "Call should be: $0 file_in file_output_prefix n"
        echo "Where *n* is used as 1/n for the probability of"
        echo "a random line to get selected."
        echo
        echo "This scripts generates two files: one called "
        echo "[file_output_prefix]_head.txt and one called "
        echo "[file_output_prefix]_rand.txt ."
        echo
        echo "They both have the same amount of lines, the "
        echo "first one is made simply by using the *head* "
        echo "command, while the second is made at random. "
        exit 2
fi

filename="$1"
fileprefix="$2"
denominator="$3"

amount_lines=$(wc -l $filename | awk '{ print $1}')
echo $(($amount_lines-1))

filename_output_rand="$fileprefix"
filename_output_rand+="_rand.txt"
filename_output_head="$fileprefix"
filename_output_head+="_head.txt"

echo "Removing previous output files, if they exist."
rm "$filename_output_head" "$filename_output_rand"

i=0
echo "Making the random file.."
while read -r line
do
        rng_value=$(( $RANDOM % $denominator))
        if [[ $rng_value -eq "0" ]]; then
                echo $line >> "$filename_output_rand"
                i=$(($i+1))
        fi

done < "$filename"
echo "Done."
echo
echo "Making the head file.."
head -n $i "$filename" >> "$filename_output_head"
echo "Done. Exiting."
