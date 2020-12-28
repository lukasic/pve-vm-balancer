#!/bin/bash

find /etc/pve/nodes/ -mindepth 3 \
    | grep qemu-server \
    | tr '/' ' ' | tr '.' ' ' \
    | awk '{ print $4, $6 }'
