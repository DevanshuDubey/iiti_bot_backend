import pathway as pw
from Pipeline.pipeline import Pipeline
from pathway.xpacks.llm import llms


## rate limit issue 
bot = llms.LiteLLMChat(
    model="groq/llama3-70b-8192"
)

class CustomServer(pw.xpacks.llm.servers.BaseRestServer):
    def __init__(
        self,
        host: str,
        port: int,
        pipeline: "Pipeline",
        with_cors: bool = False,
        **rest_kwargs,
    ):
    
        self.webserver = pw.io.http.PathwayWebserver(
            host=host,
            port=port,
            with_cors=with_cors
        )

        self.serve(
            route="/v1/chat",
            schema=pipeline.QuerySchema,
            handler=pipeline.run,
            **rest_kwargs,
        )

 
server = CustomServer(
    host="0.0.0.0", port=3000, pipeline=Pipeline(bot),
    with_cors=True
)

server.run()