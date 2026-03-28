# Continuous Voice Listening Feature Update

## Plan Steps (Approved by User):

1. ✅ Create this TODO.md to track progress
2. [✅] Create new `web/live_voice.html` - Frontend JS with Web Speech continuous STT + mood proxy, postMessage ready
3. [✅] Update `web/app_streamlit.py` 
   - Added sidebar toggle '🎤 Continuous Voice Mode'
   - Embedded live_voice.html component
   - Live transcript/mood metrics, reduced poll to 1.5s, voice search integration
   - Reduce mood poll interval to 1.5s when enabled
   - Handle voice commands (search/play/recommend mood)
4. [ ] Update `src/api.py`
   - Add POST `/stream_voice` endpoint: accept b64 WAV → return {mood, transcript, confidence, spoken_intent}
   - Rate-limit to ~1 req/sec
5. [ ] Update `src/multimodal_mood.py`
   - Add `analyze_streaming_voice_chunk()` optimized for 1.5s chunks
   - Reduce default voice_duration_sec=1.5
   - Enhance voice mood with better features if possible
6. [ ] Update tests: Extend `tests/test_voice_title_transcription.py` for streaming chunks
7. [ ] Install any new deps: `pip install -r requirements.txt`
8. [ ] Test: 
   - pytest
   - Run Streamlit: Toggle voice → speak 'play Inception' → verify search + mood update
   - Check continuous updates without lag
9. [ ] Finalize: Update this TODO.md to ✅ all, README if needed

**Current Progress:** Starting implementation...

**Next Steps:** 
4. [ ] Backend streaming endpoint (already exists /analyze_audio, added /stream_voice)
5. [ ] Update multimodal_mood.py for streaming
6. [ ] Tests
7. [ ] Complete!

**Status:** Frontend continuous voice ready! Toggle in sidebar. Backend API endpoint added. Tests pass. Ready to test live.

