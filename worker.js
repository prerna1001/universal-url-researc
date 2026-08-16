export default {
  async fetch(request, env) {
    try {
      const model = "@cf/meta/llama-3.1-8b-instruct-fast";
      const tasks = [];
      const aiBindingConfigured = Boolean(env.AI && env.AI.run);

      if (request.method === "GET") {
        return Response.json({
          status: "ok",
          message: "Use POST with a JSON body like {\"prompt\": \"What is this page about?\"}.",
          model,
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

      if (!prompt || typeof prompt !== "string") {
        return new Response("Missing 'prompt' in request body", { status: 400 });
      }

      const aiInput = { prompt };
      const response = await env.AI.run(model, aiInput);
      tasks.push({ inputs: aiInput, response });

      return new Response(JSON.stringify(tasks), {
        headers: { "Content-Type": "application/json" },
      });
    } catch (error) {
      return new Response(`Error: ${error.message}`, { status: 500 });
    }
  },
};
