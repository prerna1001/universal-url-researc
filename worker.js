export default {
  async fetch(request, env) {
    try {
      const llmModel = "@cf/meta/llama-3.1-8b-instruct-fast";
      const embeddingModel = "@cf/baai/bge-small-en-v1.5";
      const aiBindingConfigured = Boolean(env.AI && env.AI.run);

      if (request.method === "GET") {
        return Response.json({
          status: "ok",
          message:
            "Use POST with {\"prompt\": \"...\"} for answers or {\"task\": \"embed\", \"texts\": [\"...\"]} for embeddings.",
          llmModel,
          embeddingModel,
          aiBindingConfigured,
        });
      }

      if (request.method !== "POST") {
        return new Response("Method not allowed. Use POST.", { status: 405 });
      }

      if (!aiBindingConfigured) {
        throw new Error("AI binding is not properly configured.");
      }

      let body;
      try {
        body = await request.json();
      } catch {
        return new Response("Invalid JSON body. Send {\"prompt\": \"...\"}.", {
          status: 400,
        });
      }

      const { prompt } = body || {};
      const isEmbeddingTask = body?.task === "embed";

      if (isEmbeddingTask) {
        const texts = Array.isArray(body?.texts)
          ? body.texts.map((value) => String(value || "").trim()).filter(Boolean)
          : [];

        if (!texts.length) {
          return new Response("Missing non-empty 'texts' array in request body", {
            status: 400,
          });
        }

        const response = await env.AI.run(embeddingModel, {
          text: texts,
          pooling: "mean",
        });

        return Response.json({
          task: "embed",
          model: embeddingModel,
          shape: response?.shape,
          data: response?.data || [],
          pooling: response?.pooling || "mean",
        });
      }

      if (!prompt || typeof prompt !== "string") {
        return new Response("Missing 'prompt' in request body", { status: 400 });
      }

      const aiInput = { prompt };
      const response = await env.AI.run(llmModel, aiInput);

      return new Response(JSON.stringify([{ inputs: aiInput, response }]), {
        headers: { "Content-Type": "application/json" },
      });
    } catch (error) {
      return new Response(`Error: ${error.message}`, { status: 500 });
    }
  },
};
