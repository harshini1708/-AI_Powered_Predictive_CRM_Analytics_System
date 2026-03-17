# GitHub Actions CI/CD Workflows

This directory contains automated workflows for testing, building, and deploying the I-LEAD AMS system.

## Workflows

### 1. Test (`test.yml`)

**Trigger**: Push to main/develop/claude/* branches or Pull Request to main/develop

**What it does**:
- Sets up Node.js 18
- Installs dependencies
- Runs ESLint (code style check)
- Executes full test suite with coverage
- Uploads coverage to Codecov
- Comments on PR with coverage results

**Status Badge**: ![Test](../../.github/badges/test.svg)

### 2. Build (`build.yml`)

**Trigger**: Push to main/develop/claude/* branches when backend changes

**What it does**:
- Sets up Docker Buildx
- Logs in to GitHub Container Registry (ghcr.io)
- Builds Docker image with metadata tags
- Pushes to container registry
- Caches layers for faster builds
- Comments on PR with image details

**Tags Generated**:
- `latest` - Latest default branch
- Branch name - For feature branches
- Semver - For releases (v1.2.3)
- Short SHA - Commit reference (abc1234)

### 3. Deploy (`deploy.yml`)

**Trigger**:
- Published release
- Manual workflow dispatch

**Environments**:
- `staging` - Test deployment
- `production` - Production deployment

**What it does**:
- Pulls latest code from repository
- Pulls Docker image from registry
- Runs database migrations
- Deploys services with docker-compose
- Performs health check
- Notifies deployment status

**Required Secrets**:
- `DEPLOY_KEY` - SSH private key for deployment server
- `DEPLOY_HOST` - Deployment server hostname
- `DEPLOY_USER` - SSH user for deployment
- `DEPLOY_PATH` - Path to application on server

### 4. Security (`security.yml`)

**Trigger**:
- Push to main/develop/claude/* branches
- Pull requests to main/develop
- Scheduled daily at 2 AM UTC

**Checks**:
- **Dependency Scanning**: npm audit, Snyk
- **SAST**: ESLint security rules
- **Container Scanning**: Trivy vulnerability scanner
- **Code Quality**: SonarCloud analysis

**Required Secrets** (optional):
- `SNYK_TOKEN` - Snyk API token
- `SONAR_TOKEN` - SonarCloud token

## Setup Instructions

### Initial Configuration

1. **Configure Secrets** (Settings → Secrets and variables → Actions)

   For deployment workflows:
   ```bash
   DEPLOY_KEY         # SSH private key (usually id_rsa content)
   DEPLOY_HOST        # e.g., api.example.com
   DEPLOY_USER        # e.g., deploy
   DEPLOY_PATH        # e.g., /app/ilead-ams
   ```

   For security scanning:
   ```bash
   SNYK_TOKEN         # Get from https://snyk.io
   SONAR_TOKEN        # Get from https://sonarcloud.io
   ```

2. **Configure Environments** (Settings → Environments)

   Create `staging` and `production` environments with their own secrets if needed.

3. **Set Repository Settings**:
   - Branch protection for main and develop
   - Require status checks to pass before merge
   - Require PR reviews

### Monitoring

**View Workflow Status**:
- GitHub Actions tab in repository
- Check runs on PR
- Release deployments dashboard

**Badges**:
Add to README.md:

```markdown
![Test](https://github.com/owner/repo/actions/workflows/test.yml/badge.svg)
![Build](https://github.com/owner/repo/actions/workflows/build.yml/badge.svg)
![Deploy](https://github.com/owner/repo/actions/workflows/deploy.yml/badge.svg)
```

## Workflow Patterns

### Feature Development

1. Create feature branch: `git checkout -b feature/my-feature`
2. Push commits: `git push origin feature/my-feature`
3. → **Test** workflow runs automatically
4. Create pull request
5. → **Test** and **Build** workflows run
6. Review and approve
7. Merge to develop
8. → Workflows run on develop

### Release Process

1. Create release with version tag: `git tag v1.2.3`
2. Push tag: `git push origin v1.2.3`
3. Create GitHub Release
4. → **Deploy** workflow runs automatically
5. Deploys to production after release published

### Manual Deployment

If needed to deploy without release:
1. Go to Actions → Deploy → Run workflow
2. Select environment (staging/production)
3. Select branch or tag
4. Click "Run workflow"

## Environment Variables

Configure in workflow files or as organization/repository variables:

```yaml
REGISTRY: ghcr.io
IMAGE_NAME: ${{ github.repository }}/ilead-ams-backend
```

## Error Handling

### Test Failures
- Check logs in Actions tab
- Review coverage report
- Fix issues and push new commits

### Build Failures
- Docker build errors shown in logs
- Verify Dockerfile syntax
- Check docker-compose.yml validity

### Deployment Failures
- Verify deployment credentials
- Check server connectivity
- Review migration logs
- Verify environment variables on server

## Best Practices

1. **Commit Messages**: Use clear, descriptive messages
2. **Branch Names**: Follow naming convention (feature/, bugfix/, hotfix/)
3. **PR Reviews**: Ensure code review before merge
4. **Testing**: Run tests locally before pushing
5. **Releases**: Create releases for production deployments
6. **Monitoring**: Check deployment status after merge

## Troubleshooting

### Workflow Not Triggering

- Check branch name matches trigger conditions
- Verify workflow file syntax (yml formatting)
- Check `.github/workflows/` directory
- Ensure workflow is not disabled

### Build Taking Too Long

- Docker layer caching not working
- Check `cache-from` and `cache-to` configuration
- Clear cache if needed: Actions settings → Caches

### Deployment Failing

- SSH credentials incorrect or expired
- Check `DEPLOY_PATH` exists and is readable
- Verify `.env` file is present on server
- Check docker-compose.prod.yml is valid

## Related Documentation

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Action](https://github.com/docker/build-push-action)
- [Node.js Setup Action](https://github.com/actions/setup-node)
- [SonarCloud Setup](https://docs.sonarcloud.io/)

## Support

For workflow issues:
1. Check logs in Actions tab
2. Review workflow file syntax
3. Verify secrets are configured
4. Check external service status (Snyk, SonarCloud, etc.)
