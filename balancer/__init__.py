"""Load-aware balancer in front of the vLLM replicas of one model (issue #31).

LiteLLM's ``least-busy`` herds concurrent requests onto one deployment: ties go to
the first deployment, the in-flight count is bumped only after selection, and the
count ignores how heavy each request is. This proxy picks a replica from what vLLM
itself reports (queue, running, KV-cache use), reserves the pick synchronously, and
breaks ties at random. LiteLLM sees one ``api_base`` per model — this proxy.
"""
