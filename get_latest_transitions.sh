wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1g6KB5t3QUj7FLmROGOWb_w7cxOHNzkyR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g6KB5t3QUj7FLmROGOWb_w7cxOHNzkyR" -O transitions_433581_15000.npy && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1K_Va61bDrF6RlEGgZysOVPg8kJXWs6fS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1K_Va61bDrF6RlEGgZysOVPg8kJXWs6fS" -O VAE.zip && rm -rf /tmp/cookies.txt

unzip VAE.zip
