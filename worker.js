export default {
  async fetch(request, env) {
    try {
      const tasks = [];

      // Check if AI binding is configured
      if (!env.AI || !env.AI.run) {
        throw new Error("AI binding is not properly configured.");
      }

      // Read prompt from request body sent by your app
      const body = await request.json();
      const { prompt } = body || {};

      if (!prompt || typeof prompt !== "string") {
        return new Response("Missing 'prompt' in request body", { status: 400 });
      }

      const aiInput = { prompt };
      const response = await env.AI.run("@cf/meta/llama-3-8b-instruct", aiInput);
      tasks.push({ inputs: aiInput, response });

      // Return response
      return new Response(JSON.stringify(tasks), {
        headers: { "Content-Type": "application/json" },
      });
    } catch (error) {
      return new Response(`Error: ${error.message}`, { status: 500 });
    }
  },
};