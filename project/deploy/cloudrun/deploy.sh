#!/usr/bin/env bash
# Google Cloud Run: API service + daily pipeline as a Cloud Run Job triggered by Cloud Scheduler.
# TEMPLATE, never run by CI. Read it, set the variables, run it step by step.
set -euo pipefail
: "${PROJECT_ID:?}" "${REGION:=us-east1}" "${OWNER:?github owner, lowercase}"
SA="driftrev-run@${PROJECT_ID}.iam.gserviceaccount.com"
AR="${REGION}-docker.pkg.dev/${PROJECT_ID}/ghcr-remote/${OWNER}/driftrev:latest"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com cloudscheduler.googleapis.com

# 1. image: Artifact Registry remote repository proxying ghcr.io (public package) or copy it with `docker pull/tag/push`
gcloud artifacts repositories create ghcr-remote --repository-format=docker --location="$REGION" \
  --mode=remote-repository --remote-docker-repo=https://ghcr.io || true

# 2. storage for /data (GCS FUSE) and secrets
gcloud storage buckets create gs://driftrev-data --location="$REGION" || true
gcloud iam service-accounts create driftrev-run || true
gcloud storage buckets add-iam-policy-binding gs://driftrev-data --member="serviceAccount:$SA" --role=roles/storage.objectUser
printf '%s' "$(openssl rand -hex 32)" | gcloud secrets create driftrev-api-token --data-file=- || true
printf '%s' "Your Name you@example.com" | gcloud secrets create sec-user-agent --data-file=- || true
for s in driftrev-api-token sec-user-agent; do
  gcloud secrets add-iam-policy-binding "$s" --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor
done

# 3. API service (edit PROJECT_ID/REGION/OWNER in service.yaml first, or use sed)
sed -e "s/PROJECT_ID/$PROJECT_ID/g" -e "s/REGION/$REGION/g" -e "s/OWNER/$OWNER/g" deploy/cloudrun/service.yaml > /tmp/driftrev-service.yaml
gcloud run services replace /tmp/driftrev-service.yaml --region "$REGION"

# 4. daily pipeline as a Cloud Run Job sharing the same bucket (run it while the API is idle: one writer)
gcloud run jobs deploy driftrev-daily --region "$REGION" --image "$AR" --service-account "$SA" \
  --command python --args=-m,driftrev.pipeline,--mode,update --memory 4Gi --cpu 2 --task-timeout 3600 \
  --set-env-vars DRIFTREV_DATA_DIR=/data --set-secrets SEC_USER_AGENT=sec-user-agent:latest \
  --add-volume name=data,type=cloud-storage,bucket=driftrev-data --add-volume-mount volume=data,mount-path=/data

# 5. Cloud Scheduler: 22:30 New York time, Mon-Fri
gcloud scheduler jobs create http driftrev-daily --location "$REGION" --schedule "30 22 * * 1-5" \
  --time-zone America/New_York --http-method POST \
  --uri "https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/driftrev-daily:run" \
  --oauth-service-account-email "$SA"

echo "API URL:"; gcloud run services describe driftrev-api --region "$REGION" --format='value(status.url)'
