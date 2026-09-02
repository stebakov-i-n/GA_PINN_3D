# echo "=== Disk cleanup ==="
# df -h
# rm -rf /root/.clearml/venvs-cache/* 2>/dev/null || true
# rm -rf /root/.cache/pip/* 2>/dev/null || true
# pip cache purge 2>/dev/null || true
# apt-get clean
# echo "=== Available space after cleanup==="
df -h
# apt-get install -y git
# apt-get install -y libgl1

echo "=== Clearing stale git credentials / vcs cache ==="
git config --global --unset-all credential.helper 2>/dev/null || true
rm -f /root/.git-credentials 2>/dev/null || true
rm -rf /root/.clearml/vcs-cache/GA_PINN_3D.git.* 2>/dev/null || true