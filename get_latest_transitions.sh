wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1U0MzslsZU5XiHu93k-fwZgtHuJ3GXtJY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1U0MzslsZU5XiHu93k-fwZgtHuJ3GXtJY" -O transitions_121034.npy && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1K_Va61bDrF6RlEGgZysOVPg8kJXWs6fS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1K_Va61bDrF6RlEGgZysOVPg8kJXWs6fS" -O VAE.zip && rm -rf /tmp/cookies.txt

unzip VAE.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cC5cSaT7pmwQS3jCth2FrMgQes2go8JM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cC5cSaT7pmwQS3jCth2FrMgQes2go8JM" -O VAE_1080.zip && rm -rf /tmp/cookies.txt

unzip VAE_1080.zip
