# Movie Recommender - Full Feature Restoration
Generated: by BLACKBOXAI - Restore all original features (training, data, scanning/UI)

## Restoration Status
**Goal:** Undo recent simplifications, restore full multimodal UI, scraper service, training pipelines.

### Approved Restoration Plan Steps:

**Phase 1: Core Files Restore (Priority)**
- [✅] Step 1: Restore web/app_streamlit.py from backup (.bak-20260315-2217) - full voice/mood UI
- [✅] Step 2: Restore src/scraper_service.py - periodic TMDB scraper
- [✅] Step 3: Create src/multimodal_mood.py - voice + emotion backend
- [✅] Step 4: Verify web/live_voice.html, web/live_mood.html work with UI (live_voice.html present)

**Phase 2: Verify Training & Data**
- [ ] Step 5: Test emotion training: python -m src.train_emotion_model
- [ ] Step 6: Test personalization: python -m src.personalization_model train
- [ ] Step 7: Rebuild caches: python -m src.recomender build_index

**Phase 3: Integrations & Services**
- [ ] Step 8: Update src/api.py to use restored scraper_service
- [ ] Step 9: Test full stack: ./start_services.sh
- [ ] Step 10: Run tests: ./run_pytest.sh

**Phase 4: Validation**
- [ ] Step 11: streamlit run web/app_streamlit.py → test mood/voice/scraper
- [ ] Step 12: Check logs: tail -f logs/scraper.log logs/streamlit.log

**Progress: 0/12 steps complete**

Next: Implement step-by-step, updating this file after each completion.

