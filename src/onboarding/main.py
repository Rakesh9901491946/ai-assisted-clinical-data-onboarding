import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("onboarding.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

