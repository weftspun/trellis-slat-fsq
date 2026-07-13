// Erlang NIF wrapping the Slang-compiled FSQ encode kernel (CPU target).
//
// Build (see build_windows.ps1): slangc priv/slang/fsq_nif.slang -target cpp -> fsq_nif.gen.cpp,
// then cl /LD this file -> priv/fsq_nif.dll. The generated code + Slang prelude provide
// EntryPointParams_0 and the exported fsq_encode(ComputeVaryingInput*, void*, void*) group dispatcher
// ([numthreads(256,1,1)] -> we launch ceil(n/256) groups).
//
// NIF API (called from TrellisSlatFsq.SlangPort.Nif):
//   encode_raw(latent_f32_binary, n, d, levels_list, basis_list) -> indices_s32_binary
//   loaded() -> true   (stub returns false until the NIF is loaded)

#include "fsq_nif.gen.cpp"

#include <erl_nif.h>
#include <vector>

static bool get_int_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<int32_t>& out)
{
    unsigned len;
    if (!enif_get_list_length(env, list, &len))
        return false;
    out.resize(len);
    ERL_NIF_TERM head, tail = list;
    for (unsigned i = 0; i < len; ++i)
    {
        int v;
        if (!enif_get_list_cell(env, tail, &head, &tail) || !enif_get_int(env, head, &v))
            return false;
        out[i] = v;
    }
    return true;
}

static ERL_NIF_TERM encode_raw(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    ErlNifBinary latent;
    unsigned n, d;
    std::vector<int32_t> levels, basis;

    if (argc != 5 || !enif_inspect_binary(env, argv[0], &latent) ||
        !enif_get_uint(env, argv[1], &n) || !enif_get_uint(env, argv[2], &d) ||
        !get_int_list(env, argv[3], levels) || !get_int_list(env, argv[4], basis) ||
        levels.size() != d || basis.size() != d || latent.size != (size_t)n * d * sizeof(float))
        return enif_make_badarg(env);

    ErlNifBinary out;
    if (!enif_alloc_binary((size_t)n * sizeof(int32_t), &out))
        return enif_make_badarg(env);

    EntryPointParams_0 params;
    params.latent_0.data = (float*)latent.data;
    params.latent_0.count = (size_t)n * d;
    params.levels_0.data = levels.data();
    params.levels_0.count = d;
    params.basis_0.data = basis.data();
    params.basis_0.count = d;
    params.outIndex_0.data = (int32_t*)out.data;
    params.outIndex_0.count = n;
    params.n_0 = n;
    params.d_0 = d;

    ComputeVaryingInput vi;
    vi.startGroupID = {0, 0, 0};
    vi.endGroupID = {(n + 255) / 256, 1, 1};
    fsq_encode(&vi, &params, nullptr);

    return enif_make_binary(env, &out);
}

static ERL_NIF_TERM loaded(ErlNifEnv* env, int, const ERL_NIF_TERM[])
{
    return enif_make_atom(env, "true");
}

static ErlNifFunc nif_funcs[] = {
    {"encode_raw", 5, encode_raw, 0},
    {"loaded?", 0, loaded, 0},
};

ERL_NIF_INIT(Elixir.TrellisSlatFsq.SlangPort.Nif, nif_funcs, NULL, NULL, NULL, NULL)
