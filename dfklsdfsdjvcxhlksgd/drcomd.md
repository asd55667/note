#!/bin/sh

#校园网断线重连
```
while true; do
    ping www.baidu.com -c1 > /dev/null 2>&1 || wget http://172.16.8.70/0.htm --post-data="DDDDD=186141118&upass=812683&v46s=0&v6ip=&f4serip=172.30.201.2&0MKKey=" --delete-after
    sleep 10
done
```