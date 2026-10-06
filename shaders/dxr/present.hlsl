cbuffer Constants:register(b0){uint source;float exposure;uint tone;uint unused;}
struct Vertex {float4 position:SV_Position;float2 uv:TEXCOORD0;};
Vertex VS(uint id:SV_VertexID){Vertex v;v.uv=float2((id<<1)&2,id&2);v.position=float4(v.uv*float2(2,-2)+float2(-1,1),0,1);return v;}
float4 PS(Vertex input):SV_Target {
    Texture2D<float4> image=ResourceDescriptorHeap[source];SamplerState linearClamp=SamplerDescriptorHeap[8];
    float3 c=image.SampleLevel(linearClamp,input.uv,0).xyz;c=max(0,select(isfinite(c),c,0))*exp2(exposure);
    if(tone==1)c=c/(1+c);if(tone==2)c=saturate((c*(2.51*c+.03))/(c*(2.43*c+.59)+.14));
    return float4(saturate(select(c<=.0031308,12.92*c,1.055*pow(max(0,c),1/2.4)-.055)),1);
}
