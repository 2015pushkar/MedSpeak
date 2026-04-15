import { getBackendApiUrl } from "@/lib/server/backend";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const response = await fetch(`${getBackendApiUrl()}/api/analyze`, {
      method: "POST",
      body: formData,
      cache: "no-store",
    });
    const body = await response.text();
    return new Response(body, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") ?? "application/json",
      },
    });
  } catch {
    return Response.json(
      {
        code: "backend_unavailable",
        message: "The MedSpeak backend could not be reached from the frontend.",
      },
      { status: 503 },
    );
  }
}
