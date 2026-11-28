# 협업 규칙

## 브랜치
- `main`에 직접 push하지 않습니다. 브랜치에서 작업하고 PR로 합칩니다.
- 브랜치 이름은 `영역/작업`으로 짓습니다. 예: `web/upload-screen`, `recommend/engine`
- 작업을 시작하기 전에 `main`을 최신으로 받습니다.

```powershell
git switch main; git pull
git switch -c recommend/engine
```

## 커밋 메시지
`[영역] 한 줄 설명`으로 씁니다. 영역은 `model`, `web`, `recommend`, `contract`, `data`, `docs` 중 하나입니다.

예: `[web] 결과 화면에 흉터 박스 표시`

## PR
- 자기가 맡은 작업 범위만 고칩니다. 다른 사람이 맡은 파일을 고쳤다면, 그 사람의 확인을 받은 뒤 합칩니다.
- 합칠 때는 Squash and merge를 쓰고, 합친 뒤 브랜치를 지웁니다.
- 합치기 전에 아래 검사를 통과해야 합니다.

```powershell
node contract/check.ts
node --test "recommend/*.test.ts"
```
`web/`이 생기면 `npm run build`도 통과해야 합니다.

## 계약(`contract/`) 변경
- 바꾸기 전에 팀 채팅에 먼저 알립니다.
- 계약 변경은 다른 작업과 섞지 않고 PR 하나로 따로 올립니다.
- 다른 팀원 한 명이 확인한 뒤 합칩니다.

## 올리지 않는 것
- 얼굴 사진과 시연용 사진
- API 키와 `.env` 파일. 키는 Vercel 환경변수에만 넣습니다.
- 병원 홈페이지 캡처. 공유 드라이브에 둡니다.
- `node_modules/`, `.next/`
- `.onnx` 파일을 Git LFS로 올리지 않습니다. Vercel이 LFS 파일을 받지 못합니다.

키를 실수로 올렸다면 바로 폐기하고 새로 발급합니다.

## 저장소 받기
웹만 작업한다면 가중치 없이 받습니다.

```powershell
$env:GIT_LFS_SKIP_SMUDGE=1; git clone https://github.com/rhdwogml/acne-scar-2stage
```

## LLM 사용
- 작업을 시작할 때 LLM에게 이 문서와 `contract/README.md`, `contract/types.ts`를 먼저 읽게 합니다.
- LLM이 만든 코드는 직접 실행해 보고 올립니다.
- API 키를 LLM 대화창에 붙여 넣지 않습니다.
