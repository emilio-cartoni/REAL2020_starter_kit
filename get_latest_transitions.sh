wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QjPEPI2MUCTlPQBr_laWIj6jHBvoOpz0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QjPEPI2MUCTlPQBr_laWIj6jHBvoOpz0" -O best_transitions.zip && rm -rf /tmp/cookies.txt
unzip best_transitions.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_0t7P8OTa1u7CHZ5POv4GzTYeh_6WAg_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_0t7P8OTa1u7CHZ5POv4GzTYeh_6WAg_" -O VAE.zip && rm -rf /tmp/cookies.txt

unzip VAE.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eTjfLaqzDR4NqCpp1MngaAISjfBT6Hjq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eTjfLaqzDR4NqCpp1MngaAISjfBT6Hjq" -O VAE_1080.zip && rm -rf /tmp/cookies.txt

#unzip VAE_1080.zip
