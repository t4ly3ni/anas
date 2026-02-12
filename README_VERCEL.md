Deploying this Streamlit app to Vercel (Docker builder)
-----------------------------------------------------

Important: Vercel is optimized for serverless frontends. Streamlit is a long-running web process. Vercel can build Docker images via the `@vercel/docker` builder, but this approach may encounter platform limits (memory, cold starts, timeouts). For reliable hosting of a Streamlit app consider Streamlit Cloud, Render, Fly.io, Railway, or a VPS/container host. If you still want to try Vercel, follow these steps.

1) Ensure the repository includes the model artifacts and `artifacts/` directory, or adjust the Dockerfile to fetch them during build (CI) — otherwise the container will fail at runtime.

2) Make sure `vercel.json` is present (it is) and points to the `Dockerfile` (project root).

3) From your project folder, login and deploy with the Vercel CLI:

```bash
# install vercel CLI if you don't have it
npm i -g vercel

cd detection_car_price
vercel login
vercel --prod
```

4) If Vercel asks for a framework preset, choose `Other` or `Docker`.

Notes & troubleshooting
- If the container fails to start, check Vercel build and runtime logs. Large model files included in the repo increase build time and may exceed Vercel limits.
- Consider deploying a slim API on Vercel that proxies requests to a Streamlit app hosted on Render/Fly/Streamlit Cloud.
- If you prefer, I can prepare a Render or Fly.io deployment (both support containers easily and are better suited for Streamlit).

GitHub Actions (optional)
-------------------------

A GitHub Action is included at `.github/workflows/deploy_vercel.yml` that will run `vercel --prod` on pushes to `main`. To use it, add the following secret to your repository settings:

- `VERCEL_TOKEN` — a Vercel personal token with deploy permissions.

If you prefer to let Vercel deploy automatically from the repository (no action), you can instead connect the Git repository in the Vercel dashboard.

Build-time model download
-------------------------

If you don't want to commit large model/artifact files to the repo, you can set the `MODEL_ARCHIVE_URL` build argument in Vercel (under Environment > Build & Development) to a URL pointing to a zipped/tarred archive containing `models/` and `artifacts/`. The Dockerfile will download and extract it during build.

