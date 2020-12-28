#!/bin/bash

cd /var/lib/rrdcached/db/pve2-vm

for i in `ls /var/lib/rrdcached/db/pve2-vm/`; do
  rrdtool fetch /var/lib/rrdcached/db/pve2-vm/$i AVERAGE \
    --start=now-7d --end=now-1h \
    | tail -n +3 \
    | head -n -1 \
    | awk '{ print $1, '$i', $2*$3*100, $4/1024/1024, $8+$9, $10+$11 }' \
    | tr -d ':' \
    | grep -v '\-nan'
done;

