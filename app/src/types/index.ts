// API Response Types
export interface AnalyzeResponse {
  ideology_vector: number[];
  predicted_label: string;
  label_confidence: number;
  multi_label_ideology: IdeologyScores;
  important_tokens: TokenImportance[];
  neighbors?: NeighborMatch[];
  processing_info: {
    original_length: number;
    cleaned_length: number;
    model_used: string;
    embedding_dim: number;
  };
}

export interface IdeologyScores {
  social: string;
  social_confidence: number;
  economic: string;
  economic_confidence: number;
  foreign: string;
  foreign_confidence: number;
  axis_scores: number[];
}

export interface TokenImportance {
  token: string;
  importance_score: number;
  position: number;
}

export interface NeighborMatch {
  politician: string;
  similarity_score: number;
  sample_text: string;
  date?: string;
  political_lean?: string;
}

export interface CompareResponse {
  cosine_similarity: number;
  euclidean_distance: number;
  similarity_interpretation: string;
  ideology_analysis: {
    text_a_ideology: IdeologyBreakdown;
    text_b_ideology: IdeologyBreakdown;
    ideology_agreement: {
      political: boolean;
      social: boolean;
      economic: boolean;
      foreign: boolean;
    };
  };
}

export interface IdeologyBreakdown {
  political: string;
  social: string;
  economic: string;
  foreign: string;
}

export interface VisualizationResponse {
  plot_data: {
    x: number[];
    y: number[];
    colors?: string[];
    texts: string[];
    politicians?: string[];
    types?: string[];
  };
  plot_html?: string;
  clusters?: number[];
  analysis: {
    total_points?: number;
    input_points?: number;
    reference_points?: number;
    ideology_distribution?: {
      left: number;
      center: number;
      right: number;
    };
    [key: string]: any;
  };
}

// API Request Types
export interface AnalyzeRequest {
  text: string;
  include_neighbors?: boolean;
  top_k?: number;
  return_attention?: boolean;
}

export interface CompareRequest {
  text_a: string;
  text_b: string;
}

export interface VisualizationRequest {
  texts: string[];
  method?: string;
  color_by?: string;
  include_references?: boolean;
} 