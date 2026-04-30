# echo "=== Disk cleanup ==="
# rm -rf /root/.clearml/venvs-cache/* 2>/dev/null || true
# rm -rf /root/.cache/pip/* 2>/dev/null || true
# pip cache purge 2>/dev/null || true
# apt-get clean
# echo "=== Available space after cleanup==="
# df -h
# apt-get install -y libgl1