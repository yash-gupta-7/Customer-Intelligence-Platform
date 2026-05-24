// Customer Intelligence Platform — Azure Container Apps + ACR + Static Web App
targetScope = 'resourceGroup'

@description('Short name prefix for resources (letters/numbers only for ACR)')
param baseName string = 'cip'

@description('Azure region')
param location string = resourceGroup().location

@description('Container image (ACR login server + repo:tag)')
param containerImage string

@description('API secret for X-API-Key')
@secure()
param apiSecretKey string

@description('Comma-separated CORS origins')
param allowedOrigins string = '*'

@description('Deploy Azure Static Web App for frontend')
param deployStaticWebApp bool = true

@description('Deploy the API Container App (set false for ACR-only bootstrap)')
param deployApi bool = true

var acrName = toLower('${baseName}acr${uniqueString(resourceGroup().id)}')
var logName = '${baseName}-logs'
var envName = '${baseName}-env'
var appName = '${baseName}-api'
var swaName = '${baseName}-web'

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource containerEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

var acrUser = acr.listCredentials().username
var acrPass = acr.listCredentials().passwords[0].value

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = if (deployApi) {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acrUser
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acrPass
        }
        {
          name: 'api-secret-key'
          value: apiSecretKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          resources: {
            cpu: json('2.0')
            memory: '4Gi'
          }
          env: [
            {
              name: 'APP_ENV'
              value: 'production'
            }
            {
              name: 'API_SECRET_KEY'
              secretRef: 'api-secret-key'
            }
            {
              name: 'ALLOWED_ORIGINS'
              value: allowedOrigins
            }
            {
              name: 'LLM_PROVIDER'
              value: 'local'
            }
            {
              name: 'MLFLOW_TRACKING_URI'
              value: 'http://localhost:5000'
            }
            {
              name: 'LOG_LEVEL'
              value: 'INFO'
            }
            {
              name: 'UVICORN_WORKERS'
              value: '1'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 60
              periodSeconds: 30
              failureThreshold: 3
            }
            {
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              failureThreshold: 30
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 2
      }
    }
  }
}

resource staticSite 'Microsoft.Web/staticSites@2023-01-01' = if (deployStaticWebApp) {
  name: swaName
  location: location
  sku: {
    name: 'Free'
    tier: 'Free'
  }
  properties: {}
}

output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
output containerAppFqdn string = deployApi ? containerApp.properties.configuration.ingress.fqdn : ''
output containerAppUrl string = deployApi ? 'https://${containerApp.properties.configuration.ingress.fqdn}' : ''
output staticWebAppHostname string = deployStaticWebApp ? staticSite.properties.defaultHostname : ''
output staticWebAppUrl string = deployStaticWebApp ? 'https://${staticSite.properties.defaultHostname}' : ''
