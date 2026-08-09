/**
 * `/api/v1/reports` — the resource client.
 *
 * The one thing worth knowing here is `409 report_not_ready`. A report row is
 * created when its investigation starts, so a client polling before the Report
 * agent has run gets a 409, not a 404. `isReportNotReady` exists so a caller can
 * tell "keep waiting" from "this will never exist" without string-matching an
 * error message.
 *
 * The shapes below mirror `backend/schemas/report.py` rather than
 * `src/types/report.ts`. Those two drifted — the type file was written from the
 * API reference and the backend schema was written from the domain model, and
 * they disagree about where confidence and citations live. This module is
 * written against what the server actually returns, and the mismatch is called
 * out in the response types so whoever reconciles them can see both.
 */
import { type ApiError, apiFetch, isApiError } from '@/lib/api/client';

export type ConfidenceBand = 'low' | 'moderate' | 'high';

export interface ReportCitation {
  id: string;
  signal_id: string;
  quote: string;
  char_start: number;
  char_end: number;
  relevance: number;
}

export interface ReportSectionItem {
  id: string;
  ordinal: number;
  heading: string;
  body: string;
  confidence: number;
  citations: ReportCitation[];
}

export interface ReportSummary {
  id: string;
  investigation_id: string;
  title: string;
  summary: string | null;
  status: 'pending' | 'complete' | 'failed';
  format: 'markdown' | 'html' | 'pdf';
  confidence: number;
  confidence_band: ConfidenceBand;
  version: number;
  is_current: boolean;
  superseded_by: string | null;
  created_at: string;
  updated_at: string;
}

export interface ReportDetail extends ReportSummary {
  sections: ReportSectionItem[];
  /** What the investigation could not establish. A top-level field, not metadata. */
  gaps: string[];
  citation_count: number;
  /** Headings whose sections carry no citation. The reader's integrity check. */
  uncited_sections: string[];
  download_url: string | null;
}

export async function getReport(id: string): Promise<ReportDetail> {
  return apiFetch<ReportDetail>(`/reports/${encodeURIComponent(id)}`, {
    cache: 'no-store',
  });
}

export async function listReportsFor(investigationId: string): Promise<ReportSummary[]> {
  return apiFetch<ReportSummary[]>('/reports', {
    query: { investigation_id: investigationId },
    cache: 'no-store',
  });
}

/** The flat citation list, for verification tooling. */
export async function getCitations(reportId: string): Promise<ReportCitation[]> {
  return apiFetch<ReportCitation[]>(`/reports/${encodeURIComponent(reportId)}/citations`);
}

/**
 * Whether an error means "the report is not written yet".
 *
 * Distinguished from a 404 because the two demand opposite responses: this one
 * means poll again, a 404 means stop. A caller that treats them the same either
 * gives up on a report seconds from existing or retries forever against one that
 * never will.
 */
export function isReportNotReady(error: unknown): error is ApiError {
  return isApiError(error) && error.status === 409;
}
