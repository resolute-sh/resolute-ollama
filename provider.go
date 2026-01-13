// Package ollama provides Ollama LLM integration activities for resolute workflows.
package ollama

import (
	"github.com/resolute-sh/resolute/core"
	"go.temporal.io/sdk/worker"
)

const (
	ProviderName    = "resolute-ollama"
	ProviderVersion = "1.0.0"
)

// Provider returns the Ollama provider for registration.
func Provider() core.Provider {
	return core.NewProvider(ProviderName, ProviderVersion).
		AddActivity("ollama.GenerateEmbedding", GenerateEmbeddingActivity).
		AddActivity("ollama.BatchEmbed", BatchEmbedActivity)
}

// RegisterActivities registers all Ollama activities with a Temporal worker.
func RegisterActivities(w worker.Worker) {
	core.RegisterProviderActivities(w, Provider())
}
