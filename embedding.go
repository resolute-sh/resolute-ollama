package ollama

import (
	"context"
	"fmt"

	"github.com/resolute-sh/resolute/core"
	transform "github.com/resolute-sh/resolute-transform"
)

const defaultModel = "nomic-embed-text"

// GenerateEmbeddingInput is the input for GenerateEmbeddingActivity.
type GenerateEmbeddingInput struct {
	BaseURL  string
	Model    string
	Document transform.Document
}

// GenerateEmbeddingOutput is the output of GenerateEmbeddingActivity.
type GenerateEmbeddingOutput struct {
	DocumentID string
	Embedding  []float64
	Dimensions int
}

// GenerateEmbeddingActivity generates an embedding for a single document.
func GenerateEmbeddingActivity(ctx context.Context, input GenerateEmbeddingInput) (GenerateEmbeddingOutput, error) {
	client := NewClient(ClientConfig{
		BaseURL: input.BaseURL,
	})

	model := input.Model
	if model == "" {
		model = defaultModel
	}

	text := input.Document.Content
	if text == "" {
		text = input.Document.Title
	}

	embedding, err := client.GenerateEmbedding(ctx, model, text)
	if err != nil {
		return GenerateEmbeddingOutput{}, fmt.Errorf("generate embedding: %w", err)
	}

	return GenerateEmbeddingOutput{
		DocumentID: input.Document.ID,
		Embedding:  embedding,
		Dimensions: len(embedding),
	}, nil
}

// BatchEmbedInput is the input for BatchEmbedActivity.
type BatchEmbedInput struct {
	BaseURL      string
	Model        string
	DocumentsRef core.DataRef
}

// EmbeddedDocument represents a document with its embedding.
type EmbeddedDocument struct {
	Document   transform.Document
	Embedding  []float64
	Dimensions int
}

// BatchEmbedOutput is the output of BatchEmbedActivity.
type BatchEmbedOutput struct {
	Ref    core.DataRef
	Count  int
	Failed int
}

// BatchEmbedActivity generates embeddings for documents from a DataRef.
func BatchEmbedActivity(ctx context.Context, input BatchEmbedInput) (BatchEmbedOutput, error) {
	docs, err := transform.LoadDocuments(ctx, input.DocumentsRef)
	if err != nil {
		return BatchEmbedOutput{}, fmt.Errorf("load documents: %w", err)
	}

	client := NewClient(ClientConfig{
		BaseURL: input.BaseURL,
	})

	model := input.Model
	if model == "" {
		model = defaultModel
	}

	storage, err := core.GetStorage()
	if err != nil {
		return BatchEmbedOutput{}, fmt.Errorf("get storage: %w", err)
	}

	embeddings := make([]EmbeddedDocument, 0, len(docs))
	failed := 0

	for _, doc := range docs {
		text := doc.Content
		if text == "" {
			text = doc.Title
		}

		if text == "" {
			failed++
			continue
		}

		embedding, err := client.GenerateEmbedding(ctx, model, text)
		if err != nil {
			failed++
			continue
		}

		embeddings = append(embeddings, EmbeddedDocument{
			Document:   doc,
			Embedding:  embedding,
			Dimensions: len(embedding),
		})
	}

	ref, err := storage.StoreJSON(ctx, SchemaEmbeddedDocument, embeddings)
	if err != nil {
		return BatchEmbedOutput{}, fmt.Errorf("store embeddings: %w", err)
	}
	ref.Count = len(embeddings)

	return BatchEmbedOutput{
		Ref:    ref,
		Count:  len(embeddings),
		Failed: failed,
	}, nil
}

// GenerateEmbedding creates a node for generating a single embedding.
func GenerateEmbedding(input GenerateEmbeddingInput) *core.Node[GenerateEmbeddingInput, GenerateEmbeddingOutput] {
	return core.NewNode("ollama.GenerateEmbedding", GenerateEmbeddingActivity, input)
}

// BatchEmbed creates a node for generating embeddings for multiple documents.
func BatchEmbed(input BatchEmbedInput) *core.Node[BatchEmbedInput, BatchEmbedOutput] {
	return core.NewNode("ollama.BatchEmbed", BatchEmbedActivity, input)
}
