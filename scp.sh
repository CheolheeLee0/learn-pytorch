scp -i ./id_container -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P 10874 -rp /Users/icheolhui/.cache/huggingface/datasets work@114.110.136.181:~/models/datasets

# sftp -i ./id_container -P 10874 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null work@114.110.136.181

# rsync --rsync-path=/usr/bin/rsync -av -e "ssh -i ./id_container -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 10874" /Users/icheolhui/.cache/huggingface/hub/ work@114.110.136.181:~/models/


# ssh -i ./id_container -p 10874 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null work@114.110.136.181

