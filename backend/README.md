# VoiceFit Backend (FastAPI MVP)

## 1) 설치
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2) 실행
```bash
uvicorn app.main:app --reload --port 8000
```

### Spotify 커버 조회(선택)
`cover_url`이 비어 있는 곡은 Spotify Search API로 자동 조회할 수 있습니다.

```bash
export SPOTIFY_CLIENT_ID=your_client_id
export SPOTIFY_CLIENT_SECRET=your_client_secret

# 추천곡 설명 고도화(OpenAI, 선택)
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_ENRICH_ENABLED=true
export OPENAI_MAX_RETRIES=2
export OPENAI_BACKOFF_BASE_SEC=1.0
export OPENAI_CACHE_TTL_SEC=300
```

## 3) ffmpeg (선택)
`webm/ogg/m4a` 등에서 `librosa` 로딩이 실패하면 `pydub + ffmpeg` 경로를 사용합니다.

- macOS (Homebrew):
```bash
brew install ffmpeg
```

- Ubuntu:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## 4) API 테스트
### Health
```bash
curl http://127.0.0.1:8000/health
```

### Analyze (실제)
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@./sample.wav" \
  -F "vocal_range_mode=male" \
  -F "allow_cross_gender=false" \
  -F "mock=false"
```

### Analyze (mock)
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@./sample.wav" \
  -F "mock=true"
```

## 5) 에러 응답 형식
모든 400/422/500 에러는 아래 형식으로 반환됩니다.

```json
{
  "error": {
    "code": "AUDIO_TOO_SHORT",
    "message": "업로드한 음성이 너무 짧습니다.",
    "hint": "최소 3초 이상 녹음한 파일을 업로드하세요."
  }
}
```
