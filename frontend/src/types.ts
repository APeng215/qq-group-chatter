export interface Attachment {
  message_id?: string; user_id?: string; nickname?: string; position?: number;
  resource_id?: string; input_index?: number; status?: string; media_type?: string;
  size_bytes?: number; duration_ms?: number;
}
export interface Trace {
  tool_events?: {tool_call_id: string; name: string; arguments: string; result: string; cached: boolean; duration_ms: number}[];
  trace_id: string; round_id?: string; attempt?: number; created_at?: string;
  component?: string; operation?: string; status?: string; parsed_action?: string;
  validation_error?: string; fallback_reason?: string; error_message?: string;
  current_user_message?: string; response_text?: string | null; reasoning_content?: string;
  final_reply?: string; search_notice?: string; duration_ms?: number;
  model?: string; thinking?: string; temperature?: number; response_format?: unknown;
  messages?: {role: string; content: string | Block[]}[];
  raw_response?: {choices?: {message?: {content?: string | null}}[]; [key: string]: unknown};
  usage?: {total_tokens?: number; prompt_tokens?: number; completion_tokens?: number; [key: string]: unknown} | null;
  image_context?: {enabled: boolean; max_images: number; sent_count: number; attachments: Attachment[]};
}
export interface Block { type: string; text?: string; resource_id?: string; input_index?: number }
export interface TraceSnapshot {traces: Trace[]; errors?: string[]; image_understanding?: {enabled: boolean; max_images_per_request: number}}
export interface Memory {id: string; content: string; scope: string; kind: string; owner_id?: string; metadata?: unknown; history?: unknown; created_at?: string; updated_at?: string}
export interface MemorySnapshot {memories: Memory[]; summary: Record<string, number>; errors?: string[]}
export interface Filters {search: string; component: string; apiStatus: string; business: string}
