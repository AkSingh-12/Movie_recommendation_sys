# Project Cleanup: Remove Unnecessary Files & Arrange Structure
[2024-10-XX Created by BLACKBOXAI]

## Steps:


- rm web/app_streamlit.py.bak-20260315-2217
- rm requirements.lock.txt

### 3. [PENDING] Remove generated dirs contents
- rm -rf movie_recommender.egg-info/
- rm -rf data/cache/*
- rm -rf logs/*

### 4. [PENDING] Verify cleanup
- Run: find . \\( -name '*.bak' -o -path '*/cache/*' -o -path '*/logs/*' -o -name '*.egg-info*' \\) | head -10
- Should show nothing.

### 5. [PENDING] Test project
- pip install -r requirements.txt
- streamlit run web/app_streamlit.py

### 6. [PENDING] Git commit
- git add . &amp;&amp; git commit -m 'Clean project: rm backups caches logs egg-info lockfile'
- Optional PR.

**Next: Approve step 2-3 deletes?**
