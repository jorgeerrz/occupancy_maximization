### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 63a9f3aa-31a8-11ec-3238-4f818ccf6b6c
begin
	using Pkg
	Pkg.activate("../Project.toml")
	using Plots, Distributions, PlutoUI, Parameters
	using Interpolations, StatsPlots, DelimitedFiles, ZipFile
end

# ╔═╡ 18382e9c-e812-49ff-84cc-faad709bc4c3
using LinearAlgebra

# ╔═╡ 11f37e01-7958-4e79-a8bb-06b92d7cb9ed
begin
	PlutoUI.TableOfContents(aside = true)
end

# ╔═╡ 0a29d50e-df0a-4f54-b588-faa6aee2e983
theme(:vibrant,titlefont = ("Computer Modern",16), legend_font_family = "Computer Modern", legend_font_pointsize = 14, guidefont = ("Computer Modern", 16), tickfont = ("Computer Modern", 14),foreground_color_border = :black)

# ╔═╡ 379318a2-ea2a-4ac1-9046-0fdfe8c102d4
interpolation = true

# ╔═╡ 40ee8d98-cc45-401e-bd9d-f9002bc4beba
md"# Useful functions"

# ╔═╡ 4ec0dd6d-9d3d-4d30-8a19-bc91600d9ec2
@with_kw mutable struct State
	θ::Float64 = 0
	w::Float64 = 0
	u::Float64 = 1
	v::Float64 = 0
	x::Float64 = 0
end

# ╔═╡ f997fa16-69e9-4df8-bed9-7067e1a5537d
function dwdt(θ,w,force,env)
	num = env.g*sin(θ) + env.α*cos(θ)*force - env.m*env.α*w^2*env.l*sin(2*θ)/2
	den = env.l*(4/3 -env.m*env.α*cos(θ)^2)
	num/den
end

# ╔═╡ 2f31b286-11aa-443e-a3dc-c021e6fc276c
function searchsortednearest(a,x,which)
	idx = searchsortedfirst(a,x)
	if (idx==1); return idx; end
	if (idx>length(a)); return length(a); end
	if (a[idx]==x); return idx; end
	if which == "position"
		#For ceiling interpolation
		if sign(x) > 0 
		#For nearest neighbour interpolation
		#if (abs(a[idx]-x) < abs(a[idx-1]-x))
	      return idx
	   else
	      return idx-1
	   end
	elseif which == "speed"
		#For ceiling interpolation
		if sign(x) < 0 
		#For nearest neighbour interpolation
		#if (abs(a[idx]-x) < abs(a[idx-1]-x))
	      return idx
	   else
	      return idx-1
	   end
	else
		#For ceiling interpolation
		#if sign(x) > 0 
		#For nearest neighbour interpolation
		if (abs(a[idx]-x) < abs(a[idx-1]-x))
	      return idx
	   else
	      return idx-1
	   end
	end
end

# ╔═╡ 4263babb-32ae-446f-b6a6-9d5451ed40cd
function entropy(pd, delta = 1E-2)
	out = 0
	if abs(sum(pd) - 1) > delta
		throw(ErrorException("Probabilites do not add up to 1"))
	else
		for i in 1:length(pd)
			if pd[i] < 0 || pd[i] > 1
				throw(ErrorException("Probabilites need to lie inside the simplex"))
			else
				if pd[i] > 0 #exclude log(0)
					out += - pd[i]*log(pd[i])
				end
			end
		end
	end
	out
end

# ╔═╡ 9396a0d1-6036-44ec-98b7-16df4d150b54
md"# H agent"

# ╔═╡ cfdf3a8e-a377-43ef-8a76-63cf78ce6a16
@with_kw struct inverted_pendulum_borders
	g::Float64 = 9.81
	l::Float64 = 1
	M::Float64 = 1
	m::Float64 = 0.1
	γ::Float64 = 0.99
	α::Float64 = 1/(M+m)
	Δt::Float64 = 0.01
	sizeu :: Int64 = 2
	sizev :: Int64 = 21
	sizew :: Int64 = 21
	sizex :: Int64 = 21
	sizeθ :: Int64 = 51
	nstates :: Int64 = sizeθ*sizew*sizeu*sizev*sizex
	nactions :: Int64 = 5
	max_a = 20
	a_s = collect(-max_a:2*max_a/(nactions-1):max_a)
	max_θ = 0.78 #pi/4
	max_w :: Float64 = 3
	max_v :: Float64 = 1/0.1
	max_x :: Float64 = 5
	#sq = uniform.^2 .*sign.(uniform)
	#sq ./= abs(uniform[1])
	θs = collect(-max_θ:2*max_θ/(sizeθ-1):max_θ)
	ws = collect(-max_w:2*max_w/(sizew-1):max_w)
	vs = collect(-max_v:2*max_v/(sizev-1):max_v)
	xs = collect(-max_x:2*max_x/(sizex-1):max_x)
	#Smaller values get more precision
	# θs = (collect(-max_θ:2*max_θ/(sizeθ-1):max_θ).^2 .* sign.(collect(-max_θ:2*max_θ/(sizeθ-1):max_θ)))/max_θ
	# ws = (collect(-max_w:2*max_w/(sizew-1):max_w).^2 .* sign.(collect(-max_w:2*max_w/(sizew-1):max_w)))/max_w
	# vs = (collect(-max_v:2*max_v/(sizev-1):max_v).^2 .* sign.(collect(-max_v:2*max_v/(sizev-1):max_v)))/max_v
	# xs = (collect(-max_x:2*max_x/(sizex-1):max_x).^2 .* sign.(collect(-max_x:2*max_x/(sizex-1):max_x)))/max_x
end

# ╔═╡ b1d3fff1-d980-4cc8-99c3-d3db7a71bf60
function real_transition(state::State,action,env::inverted_pendulum_borders)
	acc = dwdt(state.θ,state.w,action,env)
	new_th = state.θ + state.w*env.Δt #+ acc*env.Δt^2/2
	new_w = state.w + acc*env.Δt
	#According to the paper, but there is a sign error
	#acc_x = env.α*(action + env.m*env.l*(state.w^2*sin(state.θ)-acc*cos(state.θ)))
	acc_x = (4/3 * env.l * acc - env.g*sin(state.θ))/cos(state.θ)
	new_v = state.v + acc_x*env.Δt
	new_x = state.x + state.v*env.Δt #+ acc_x*env.Δt^2/2
	new_u = state.u
	if abs(new_th) >= env.max_θ 
		new_th = sign(new_th)*env.max_θ
		new_u = 1
	end
	if abs(new_x) >= env.max_x
		new_x = sign(new_x)*env.max_x
		new_u = 1
	end
	if abs(new_v) >=env.max_v
		new_v = sign(new_v)*env.max_x
	end
	if abs(new_w) >= env.max_w
		new_w = sign(new_w)*env.max_w
	end
	State(θ = new_th, w = new_w, v = new_v, x = new_x, u = new_u)
end

# ╔═╡ ffe32878-8732-46d5-b57c-9e9bb8e6dd74
function adm_actions_b(s::State, ip::inverted_pendulum_borders)
	out = ip.a_s
	ids = collect(1:ip.nactions)
	#If dead, only no acceleration
	if s.u < 2 
		out = [0]
		ids = [Int((ip.nactions+1)/2)]
	end
	out,Int.(ids)
end

# ╔═╡ 784178c3-4afc-4c65-93e1-4265e1faf817
function build_nonflat_index_b(state::State,env::inverted_pendulum_borders)
	idx_th = searchsortednearest(env.θs,state.θ,"position")
	idx_w = searchsortednearest(env.ws,state.w,"normal")
	idx_u = state.u
	idx_v = searchsortednearest(env.vs,state.v,"normal")
	idx_x = searchsortednearest(env.xs,state.x,"position")
	[idx_th,idx_w,idx_u,idx_v,idx_x]
end

# ╔═╡ fa56d939-09d5-4b63-965a-38016c957fbb
function build_index_b(state_index,env::inverted_pendulum_borders)
	Int64(state_index[1] + env.sizeθ*(state_index[2]-1) + env.sizeθ*env.sizew*(state_index[3]-1)+ env.sizeθ*env.sizew*env.sizeu*(state_index[4]-1)+ env.sizeθ*env.sizew*env.sizeu*env.sizev*(state_index[5]-1))
end

# ╔═╡ 05b8a93c-cefe-473a-9f8e-3e82de0861b2
function transition_b(state::State,action,env::inverted_pendulum_borders)
	if state.u > 1
		acc = dwdt(state.θ,state.w,action,env)
		new_th = state.θ + state.w*env.Δt #+ acc*env.Δt^2/2
		new_w = state.w + acc*env.Δt
		#acc_x = env.α*(action + env.m*env.l*(state.w^2*sin(state.θ)-acc*cos(state.θ)))
		acc_x = (4/3 *env.l * acc - env.g*sin(state.θ))/cos(state.θ)
		new_v = state.v + acc_x*env.Δt
		new_x = state.x + state.v*env.Δt #+ acc_x*env.Δt^2/2
		idx_th = searchsortednearest(env.θs,new_th,"position")
		idx_w = searchsortednearest(env.ws,new_w,"normal")
		idx_v = searchsortednearest(env.vs,new_v,"normal")
		idx_x = searchsortednearest(env.xs,new_x,"position")
		u_new = state.u
		if abs(env.θs[idx_th]) >= env.max_θ || abs(env.xs[idx_x]) >= env.max_x
			u_new = 1
		end
	else
		idx_th = searchsortednearest(env.θs,state.θ,"position")
		idx_w = searchsortednearest(env.ws,state.w,"normal")
		idx_v = searchsortednearest(env.vs,state.v,"normal")
		idx_x = searchsortednearest(env.xs,state.x,"position")		
		u_new = state.u
	end
	θ_new = env.θs[idx_th]
	w_new = env.ws[idx_w]
	v_new = env.vs[idx_v]
	x_new = env.xs[idx_x]
	[idx_th,idx_w,u_new,idx_v,idx_x],State(θ = θ_new, w = w_new, v = v_new, u = u_new, x = x_new)
end

# ╔═╡ f92e8c6d-bf65-4786-8782-f38847a7fb7a
function discretize_state(state::State,env::inverted_pendulum_borders)
	idx_th = searchsortednearest(env.θs,state.θ,"position")
	idx_w = searchsortednearest(env.ws,state.w,"normal")
	idx_v = searchsortednearest(env.vs,state.v,"normal")
	idx_x = searchsortednearest(env.xs,state.x,"position")
	[idx_th,idx_w,state.u,idx_v,idx_x],State(θ = env.θs[idx_th], w = env.ws[idx_w], u = state.u, v = env.vs[idx_v], x = env.xs[idx_x])
end

# ╔═╡ f9121267-8d7e-40b1-b9cc-c8da3f45cdb8
function reachable_states_b(state::State,action,env::inverted_pendulum_borders)
	[transition_b(state,action,env)[1]],[transition_b(state,action,env)[2]]
end

# ╔═╡ 36601cad-1ba9-48f2-8463-a58f98bedd34
function degeneracy_cost(state,env::inverted_pendulum_borders,δ = 1E-5)
	-δ*((state.θ/env.max_θ)^2+(state.x/env.max_x)^2+(state.v/env.max_v)^2+(state.w/env.max_w)^2)/4
end

# ╔═╡ ce8bf897-f240-44d8-ae39-cf24eb115704
function iteration(env::inverted_pendulum_borders, tolerance = 1E0, n_iter = 100;δ = 1E-5)
	v = zeros(env.nstates)
	v_new = zeros(env.nstates)
	t_stop = n_iter
	error = 0
	f_error = 0
	for t in 1:n_iter
		ferror_max = 0
		#Parallelization over states
		Threads.@threads for idx_θ in 1:env.sizeθ
			for idx_w in 1:env.sizew, idx_u in 1:env.sizeu, idx_v in 1:env.sizev, idx_x in 1:env.sizex
				state_idx = [idx_θ,idx_w,idx_u,idx_v,idx_x]
				state = State(θ = env.θs[idx_θ], w = env.ws[idx_w], u = idx_u, v = env.vs[idx_v], x = env.xs[idx_x])
				i = build_index_b(state_idx,env)
				actions,ids_actions = adm_actions_b(state,env)
				sum = 0
				# Add negligible external reward that breaks degeneracy for 
				# Q agent
				small_reward = degeneracy_cost(state,env,δ)
				for (id_a,a) in enumerate(actions)
					#For every action, look at reachable states
					s_primes_ids,states_p = reachable_states_b(state,a,env)
					exponent = 0
					for (idx,s_p) in enumerate(s_primes_ids)
						i_p = build_index_b(s_p,env)
						P = 1
						exponent += env.γ*P*v[i_p]
					end
					sum += exp(exponent + small_reward)
				end
				v_new[i] = log(sum)
				f_error = abs(v[i] - v_new[i])
				# Use supremum norm of difference between values
				# at different iterations
				ferror_max = max(ferror_max,f_error)
			end
		end
		# Check overall error between value's values at different iterations
		error = norm(v-v_new)/norm(v)
		#if f_error < tolerance
		if ferror_max < tolerance
			t_stop = t
			break
		end
		println("iteration = ", t, ", error = ", error, ", max function error = ", ferror_max)
		v = deepcopy(v_new)
	end
	v_new,error,t_stop
end

# ╔═╡ b975eaf8-6a94-4e39-983f-b0fb58dd70a1
function optimal_policy_b(dstate,state,value,env::inverted_pendulum_borders)
	#Check admissible actions
	actions,ids_actions = adm_actions_b(dstate,env)
	policy = zeros(length(actions))
	state_index = build_nonflat_index_b(state,env)
	id_state = build_index_b(state_index,env)
	#Value at current state
	v_state = value(state.θ,state.w,state.u,state.v,state.x)
	for (idx,a) in enumerate(actions)
		#Given action, check reachable states (deterministic environment for now)
		s_primes_ids,states_p = reachable_states_b(state,a,env)
		s_prime = real_transition(state,a,env)
		exponent = 0
		#for s_p in s_primes_ids
			P = 1
			if interpolation == true
				#interpolated value
				exponent += env.γ*P*value(s_prime.θ,s_prime.w,s_prime.u,s_prime.v,s_prime.x)
			else
				#deterministic environment
				i_p = build_index_b(s_primes_ids[1],env)
				exponent += env.γ*P*value[i_p]
			end
		#end
		#Normalize at exponent
		exponent -= v_state
		policy[idx] = exp(exponent)
	end
	#Since we are using interpolated values, policy might not be normalized, so we normalize
	#println("sum policy = ", sum(policy))
	policy = policy./sum(policy)
	#Return available actions and policy distribution over actions
	actions,policy
end

# ╔═╡ e0fdce26-41b9-448a-a86a-6b29b68a6782
function interpolate_value(flat_value,env::inverted_pendulum_borders)
	value_reshaped = reshape(flat_value,length(env.θs),length(env.ws),2,length(env.vs),length(env.xs))
	θs = -env.max_θ:2*env.max_θ/(env.sizeθ-1):env.max_θ
	ws = -env.max_w:2*env.max_w/(env.sizew-1):env.max_w
	vs = -env.max_v:2*env.max_v/(env.sizev-1):env.max_v
	xs = -env.max_x:2*env.max_x/(env.sizex-1):env.max_x
	itp = Interpolations.interpolate(value_reshaped,BSpline(Linear()))
	sitp = Interpolations.scale(itp,θs,ws,1:2,vs,xs)
	sitp
end

# ╔═╡ bbe19356-e00b-4d90-b704-db33e0b75743
ip_b = inverted_pendulum_borders(M = 1, m = 0.1,l = 1.,Δt = 0.02, sizeθ = 31, sizew = 31,sizev = 31, sizex = 31, max_θ = 0.62, max_w = 3, a_s = [-50,-10,0,10,50], max_x = 2.4, max_v = 3, nactions = 5, γ = 0.96)

# ╔═╡ b38cdb2b-f570-41f4-8996-c7e41551f374
function draw_cartpole(xposcar,xs,ys,t,xlimit,ip::inverted_pendulum_borders)
	hdim = 400
	vdim = 200
	size = 2
	verts = [(-size,-size/2),(size,-size/2),(size,size/2),(-size,size/2)]
	#Draw car
	pcar = plot(xticks = false, yticks = false,xlim = (-xlimit-0.3,xlimit+0.3),ylim = (-0.1, ip.l + 0.05), grid = false, axis = false,legend = false)
	#Plot arena
	plot!(pcar, [-xlimit-0.2,xlimit+0.2], [-0.1,-0.1], color = :black)
	plot!(pcar, [-xlimit-0.3,-xlimit-0.2], [0.2*ip.l,0.2*ip.l], color = :black)
	plot!(pcar, [xlimit+0.2,xlimit+0.3], [0.2*ip.l,0.2*ip.l], color = :black)
	plot!(pcar, [-xlimit-0.2,-xlimit-0.2], [-0.1,0.2*ip.l], color = :black)
	plot!(pcar, [xlimit+0.2,xlimit+0.2], [-0.1,0.2*ip.l], color = :black)
	#Draw pole
	plot!(pcar,xs[t],ys[t],marker = (:none, 1),linewidth = 2.5, linecolor = :black)
	#Draw cart
	plot!(pcar,xposcar[t],[0],marker = (Shape(verts),30),color = :pink)
	#draw_actions
	actions,_ = adm_actions_b(State(u = 2),ip)
	arrow_x = zeros(length(actions))
	arrow_y = zeros(length(actions))
	aux = actions
	for i in 1:length(aux)
		mult = 0.03
		arrow_x[i] = aux[i]*mult
		arrow_y[i] = 0#aux[i][2]*mult*1.3
	end
	quiver!(pcar,ones(Int64,length(aux))*xposcar[t][1],[0,0.04,0.4,0.04,0.0],quiver = (arrow_x,arrow_y),color = "black",linewidth = 2)
	scatter!(pcar,xposcar[t],[0.02],markersize = 10,color = "black")
	plot(pcar, size = (hdim,vdim),margin=4Plots.mm)
end

# ╔═╡ 79372251-157d-43a0-9560-4727fbd36ea9
function draw_cartpole_time(xposcar,xs,ys,xlimit,maxtime,step,ip::inverted_pendulum_borders)
	hdim = 500
	vdim = 200
	size = 1
	verts = [(-size,-size/2),(size,-size/2),(size,size/2),(-size,size/2)]
	#Draw car
	pcar = plot(xticks = false, yticks = false,xlim = (-xlimit-0.3,xlimit+0.3),ylim = (-0.1, ip.l + 0.05), grid = false, axis = false,legend = false)
	#Plot arena
	plot!(pcar, [-xlimit-0.2,xlimit+0.2], [-0.1,-0.1], color = :black)
	plot!(pcar, [-xlimit-0.3,-xlimit-0.2], [0.2*ip.l,0.2*ip.l], color = :black)
	plot!(pcar, [xlimit+0.2,xlimit+0.3], [0.2*ip.l,0.2*ip.l], color = :black)
	plot!(pcar, [-xlimit-0.2,-xlimit-0.2], [-0.1,0.2*ip.l], color = :black)
	plot!(pcar, [xlimit+0.2,xlimit+0.2], [-0.1,0.2*ip.l], color = :black)
	for t in 1:step:maxtime
		#Draw pole
		plot!(pcar,xs[t],ys[t],marker = (:none, 1),linewidth = 1, linecolor = :black,alpha = 0.1)
		#Draw cart
		plot!(pcar,xposcar[t],[0],marker = (Shape(verts),30),alpha = 0.1 ,color = :pink)
	end
	plot(pcar, size = (hdim,vdim),margin=6Plots.mm)
end

# ╔═╡ 10a47fa8-f235-4455-892f-9b1457b1f82c
function draw_cartpole_timeright(xposcar,xs,ys,maxtime,step,ip::inverted_pendulum_borders)
	hdim = 800
	vdim = 120
	size = 0.2
	Δx = 0.006/step
	verts = [(-size,-size/2),(size,-size/2),(size,size/2),(-size,size/2)]
	#Initialize car plot
	pcar = plot(xticks = false, yticks = false,xlim = (0,maxtime*Δx),ylim = (-0.1, ip.l + 0.05), grid = false, axis = false,legend = false)
	for t in 1:step:maxtime
		#Draw pole
		plot!(pcar,[0,xs[t][2]-xs[t][1]]+[t*Δx,t*Δx],ys[t],marker = (:none, 1),linewidth = 1.2, linecolor = :black,alpha = 0.3)
		#Draw cart
		plot!(pcar,xposcar[t] +[t*Δx],[0],marker = (Shape(verts),30),alpha = 0.5 ,color = :pink)
	end
	plot(pcar, size = (hdim,vdim),margin=3Plots.mm)
end

# ╔═╡ e099528b-37a4-41a2-836b-69cb3ceda2f5
md" ## Value iteration"

# ╔═╡ 6b9b1c38-0ed2-4fe3-9326-4670a33e7765
#Tolerance for iteration, supremum norm
tolb = 1E-3

# ╔═╡ a0b85a14-67af-42d6-b231-1c7d0c293f6e
#If value calculated, this code stores the value in a dat file
#writedlm("h_value_g_$(ip_q.γ)_nstates_$(ip_q.nstates).dat",h_value)

# ╔═╡ 08ecfbed-1a7c-43d4-ade7-bf644eeb6eee
function create_episode_b(state_0,int_value,max_t,env::inverted_pendulum_borders)
	x = 0.
	v = 0.
	xpositions = Any[]
	ypositions = Any[]
	xposcar = Any[]
	yposcar = Any[]
	thetas = Any[]
	ws = Any[]
	vs = Any[]
	us = Any[]
	a_s = Any[]
	values = Any[]
	entropies = Any[]
	all_x = Any[]
	all_y = Any[]
	state = deepcopy(state_0)
	for t in 1:max_t
		thetax = state.x - env.l*sin(state.θ)
		thetay = env.l*cos(state.θ)
		push!(xpositions,[state.x,thetax])
		push!(ypositions,[0,thetay])
		push!(xposcar,[state.x])
		push!(thetas,state.θ)
		push!(ws,state.w)
		push!(vs,state.v)
		push!(us,state.u)
		if state.u == 1
			break
		end
		ids_dstate,discretized_state = discretize_state(state,env)
		id_dstate = build_index_b(ids_dstate,env)
		push!(values,int_value(state.θ,state.w,state.u,state.v,state.x))
		actions,policy = optimal_policy_b(discretized_state,state,int_value,env)
		#Choosing the action with highest prob (empowerement style)
		#idx = findmax(policy)[2] 
		#Choosing action randomly according to policy
		push!(entropies,entropy(policy))
		idx = rand(Categorical(policy))
		action = actions[idx]
		push!(a_s,action)
		#For discretized dynamics
		#_,state_p = transition_b(state,action,env) 
		#For real dynamics
		state_p = real_transition(state,action,env)
		state = deepcopy(state_p)
	#end
	end
	xpositions,ypositions,xposcar,thetas,ws,us,vs,a_s,values,entropies
end

# ╔═╡ 21472cb5-d968-4306-b65b-1b25f522dd4d
md" ## Animation"

# ╔═╡ c5c9bc66-f554-4fa8-a9f3-896875a50627
interval = collect(-0.5:0.1:0.5).*pi/180

# ╔═╡ 370a2da6-6de0-44c0-9f70-4e676769f59b
state_0_anim = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval),u=2)

# ╔═╡ 24a4ba06-d3ae-4c4b-9ab3-3852273c2fd4
function animate_w_borders(xposcar,xs,ys,xlimit,maxt,values,entropies,ip::inverted_pendulum_borders,plot_with_values = true)
	hdim = 1200
	vdim = 600
	frac_for_cartpole = 0.6
	size = 1.5
	verts = [(-size,-size/2),(size,-size/2),(size,size/2),(-size,size/2)]
	anim = @animate for t in 1:length(xposcar)-1
		pcar = plot(xticks = false, yticks = false,xlim = (-xlimit-0.3,xlimit+0.3),ylim = (-0.1, ip.l + 0.05), grid = false, axis = false,legend = false)
		#Plot arena
		plot!(pcar, [-xlimit-0.2,xlimit+0.2], [-0.1,-0.1], color = :black)
		plot!(pcar, [-xlimit-0.3,-xlimit-0.2], [0.2*ip.l,0.2*ip.l], color = :black)
		plot!(pcar, [xlimit+0.2,xlimit+0.3], [0.2*ip.l,0.2*ip.l], color = :black)
		plot!(pcar, [-xlimit-0.2,-xlimit-0.2], [-0.1,0.2*ip.l], color = :black)
		plot!(pcar, [xlimit+0.2,xlimit+0.2], [-0.1,0.2*ip.l], color = :black)
		#Plot properties
		pacc = plot(ylabel = "value",xlim = (0,length(xposcar)),ylim = (round(minimum(values),sigdigits = 3)-0.1,round(maximum(values),sigdigits = 3)), xticks = false)
		pent = plot(xlabel = "time", ylabel = "action entropy",xlim = (0,length(xposcar)),ylim = (-0.1,log(ip.nactions)), xticks = false)
		if t <= maxt
			#excess = abs(xposcar[t][1]) - 1
			plot!(pcar,xs[t],ys[t],marker = (:none, 1),linewidth = 2, linecolor = :black)
			plot!(pcar,xposcar[t],[0],marker = (Shape(verts),30))
			plot!(pacc,values[1:t], label = false)
			plot!(pent, entropies[1:t], label = false)
		#scatter!(ptest,xs[t],ys[t],markersize = 50)
		#plot!(ptest,xticks = collect(0.5:env1.sizex+0.5), yticks = collect(0.5:env1.sizey+0.5), gridalpha = 0.8, showaxis = false, ylim=(0.5,env1.sizey +0.5), xlim=(0.5,env1.sizex + 0.5))
		end
		if plot_with_values == true
			plot(pcar,pacc,pent, layout = Plots.grid(3, 1, heights=[frac_for_cartpole,(1-frac_for_cartpole)/2,(1-frac_for_cartpole)/2]), size = (hdim,vdim),margin=6Plots.mm)
		else
			plot(pcar, size = (700,200),margin=5Plots.mm)
		end
	end
	anim
end

# ╔═╡ 63fe2f55-f5ce-4022-99ee-3bd4e14e6352
md"Produce animations? $(@bind movies CheckBox(default = false))"

# ╔═╡ 6107a0ce-6f01-4d0b-bd43-78f2125ac185
md"# Q agents (reward maximizer)"

# ╔═╡ c05d12f3-8b8a-4f34-bebc-77e376a980d0
function reachable_rewards(state,action,env::inverted_pendulum_borders,δ = 1E-5)
	#We break degeneracy by adding a small cost of being far away
	r = 1 - δ*((state.θ/env.max_θ)^2+(state.x/env.max_x)^2+(state.v/env.max_v)^2+(state.w/env.max_w)^2)/4
	if state.u == 1
		r = 0
	end
	[r]
end

# ╔═╡ e181540d-4334-47c4-b35d-99023c89a2c8
function Q_iteration(env::inverted_pendulum_borders, ϵ = 0.01, tolerance = 1E-2, n_iter = 100; δ = 1E-5)
	v = zeros(env.nstates)
	v_new = zeros(env.nstates)
	t_stop = n_iter
	error = 0
	ferror = 0
	for t in 1:n_iter
		f_error = 0
		ferror_old = 0
		Threads.@threads for idx_θ in 1:env.sizeθ
		#Threads.@spawn for idx_θ in 1:env.sizeθ,idx_w in 1:env.sizew, idx_u in 1:env.sizeu, idx_v in 1:env.sizev, idx_x in 1:env.sizex
			for idx_w in 1:env.sizew, idx_u in 1:env.sizeu, idx_v in 1:env.sizev, idx_x in 1:env.sizex
				state_idx = [idx_θ,idx_w,idx_u,idx_v,idx_x]
				state = State(θ = env.θs[idx_θ], w = env.ws[idx_w], u = idx_u, v = env.vs[idx_v], x = env.xs[idx_x])
				i = build_index_b(state_idx,env)
				v_old = deepcopy(v[i])
				#println("v before = ", v[i])
				actions,ids_actions = adm_actions_b(state,env)
				values = zeros(length(actions))
				for (id_a,a) in enumerate(actions)
					s_primes_ids,states_p = reachable_states_b(state,a,env)
					rewards = reachable_rewards(state,a,env,δ)
					#state_p = transition_b(state,a,env)[2]
					for (idx,s_p) in enumerate(s_primes_ids)
						#rewards = reachable_rewards(states_p[idx],a,env)
						i_p = build_index_b(s_p,env)
						for r in rewards
							#deterministic environment
							values[id_a] += r + env.γ*v[i_p]
						end
					end
				end
				#ϵ-greedy action selection
				v_new[i] = (1-ϵ)*maximum(values) + (ϵ/length(actions))*sum(values)
				ferror = abs(v[i] - v_new[i])
				#ferror = abs(v[i] - v_old)
				ferror_old = max(ferror_old,ferror)
			end
		end
		error = norm(v-v_new)/norm(v)
		#if f_error < tolerance
		if ferror_old < tolerance
			t_stop = t
			break
		end
		println("iteration = ", t, ", error = ", error, ", max function error = ", ferror_old)
		v = deepcopy(v_new)
	end
	v,error,t_stop
end

# ╔═╡ 31a1cdc7-2491-42c1-9988-63650dfaa3e3
function optimal_policy_q(dstate,state,value,ϵ,env::inverted_pendulum_borders,interpolation = true)
	actions,ids_actions = adm_actions_b(dstate,env)
	values = zeros(length(actions))
	policy = zeros(length(actions))
	#print("actions = ", actions)
	#state_index = build_nonflat_index_b(state,env)
	#id_state = build_index_b(state_index,env)
	for (idx,a) in enumerate(actions)
		s_primes_ids,states_p = reachable_states_b(state,a,env)
		s_prime = real_transition(state,a,env)
		rewards = reachable_rewards(state,a,env)
		#for (id_sp,s_p) in enumerate(s_primes_ids)
			for r in rewards
				if interpolation == true
					#interpolated value
					values[idx] += r + env.γ*value(s_prime.θ,s_prime.w,s_prime.u,s_prime.v,s_prime.x)
				else
					#deterministic environment
					i_p = build_index_b(s_primes_ids[1],env)
					values[idx] += r + env.γ*value[i_p]
				end
			end
		#end
	end
	#println("state = ", state)
	#println("policy = ", policy)
	#println("sum policy = ", sum(policy))
	#println("values = ", values)
	best_actions = findall(i-> i == maximum(values),values)
	#ϵ-greedy policy
	for i in 1:length(actions)
		if i in best_actions
			#There might be more than one optimal action
			policy[i] = (1-ϵ)/length(best_actions) + ϵ/length(actions)
		else
			#Choose random action with probability ϵ
			policy[i] = ϵ/length(actions)
		end
	end
	actions,policy,length(best_actions)
end

# ╔═╡ 87c0c3cb-7059-42ae-aed8-98a0ef2eb55f
ip_q = inverted_pendulum_borders(M = 1.0, m = 0.1,l = 1.,Δt = 0.02, sizeθ = 31, sizew = 31,sizev = 31, sizex = 31, a_s = [-50,-10,0,10,50], max_θ = 0.62, max_x = 2.4, max_v = 3, max_w = 3, nactions = 5, γ = 0.96)

# ╔═╡ 7c04e2c3-4157-446d-a065-4bfe7d1931fd
begin
#To calculate value, uncomment, it takes around 30 minutes for 1.8E6 states
#h_value,error,t_stop= iteration(ip_b,tolb,1200,δ = 1E-5)
#writedlm("h_value_g_$(ip_q.γ)_nstates_$(ip_q.nstates).dat",h_value)
#Read from compressed file
	# h_zip = ZipFile.Reader("h_value_g_$(ip_q.γ)_nstates_$(ip_q.nstates).dat.zip")
	# h_value = readdlm(h_zip.files[1], Float64)
	h_value = readdlm("h_value_g_$(ip_q.γ)_nstates_$(ip_q.nstates).dat")
end

# ╔═╡ 8a59e209-9bb6-4066-b6ca-70dac7da33c3
h_value_int = interpolate_value(h_value,ip_b);

# ╔═╡ 9230de54-3ee3-4242-bc34-25a38edfbb6b
begin
	max_t_h_anim = 1000
	#state0_h_anim = State(θ = 0.001, u = 2)
	xs_h_anim, ys_h_anim, xposcar_h_anim, thetas_ep_h_anim, ws_ep_h_anim, us_ep_h_anim, vs_ep_h_anim, actions_h_anim, values_h_anim, entropies_h_anim = create_episode_b(state_0_anim,h_value_int,max_t_h_anim, ip_b)
	length(xposcar_h_anim)
end

# ╔═╡ dafbc66b-cbc2-41c9-9a42-da5961d2eaa6
@bind t Slider(1:max_t_h_anim)

# ╔═╡ 2735cbf5-790d-40b4-ac32-a413bc1d530a
begin
	draw_cartpole(xposcar_h_anim,xs_h_anim,ys_h_anim,t,ip_b.max_x,ip_b)
	#savefig("cartpole_draw.pdf")
end

# ╔═╡ 2f1b6992-3082-412d-b966-2b4b278b2ed0
draw_cartpole_time(xposcar_h_anim,xs_h_anim,ys_h_anim,ip_b.max_x,200,1,ip_b)

# ╔═╡ a54788b9-4b1e-4066-b963-d04008bcc242
begin
	draw_cartpole_timeright(xposcar_h_anim,xs_h_anim,ys_h_anim,1000,1,ip_b)
	#savefig("h_frozen_movie2.pdf")
end

# ╔═╡ 14314bcc-661a-4e84-886d-20c89c07a28e
#Animation, it takes some time
if movies == true
	anim_b = animate_w_borders(xposcar_h_anim,xs_h_anim,ys_h_anim,ip_b.max_x,max_t_h_anim,values_h_anim,entropies_h_anim,ip_b,false)
end

# ╔═╡ c087392c-c411-4368-afcc-f9a104856884
if movies == true
gif(anim_b,fps = Int(1/ip_b.Δt),"episode_h_agent.gif")
end

# ╔═╡ f9b03b4d-c521-4456-b0b9-d4a301d8a813
md" ## Value iteration"

# ╔═╡ 355db008-9661-4e54-acd5-7c2c9ba3c7f5
tol = 1E-3

# ╔═╡ 564cbc7a-3125-4b67-843c-f4c74ccef51f
begin
#To calculate value, uncomment, it takes around 30 minutes for 1.8E6 states
	# ϵ_try = 0.5
	# q_value, q_error, q_stop = Q_iteration(ip_q,ϵ_try,tol,1200,δ = 1E-3)
#Otherwise, read from file
#Read from compressed file
	#q_zip = ZipFile.Reader("q_value_g_$(ip_q.γ)_nstates_$(ip_q.nstates).dat.zip")
	#q_value = readdlm(q_zip.files[1], Float64)
	q_value = readdlm("q_value_eps_0.25_nstates_$(ip_q.nstates).dat")
end

# ╔═╡ d20c1afe-6d5b-49bf-a0f2-a1bbb21c709f
#If calculated, this line writes the value function in a file
#writedlm("q_value_g_$(ip_q.γ)_nstates_$(ip_q.nstates).dat",q_value)

# ╔═╡ e9687e3f-be56-44eb-af4d-f169558de0fd
q_value_int = interpolate_value(q_value,ip_q);

# ╔═╡ e9a01524-90b1-4249-a51c-ec8d1624be5b
function create_episode_q(state_0,value,ϵ,max_t,env::inverted_pendulum_borders,interpolation = true)
	x = 0.
	v = 0.
	xpositions = Any[]
	ypositions = Any[]
	xposcar = Any[]
	yposcar = Any[]
	thetas = Any[]
	ws = Any[]
	vs = Any[]
	us = Any[]
	a_s = Any[]
	values = Any[]
	entropies = Any[]
	rewards = Any[]
	all_x = Any[]
	all_y = Any[]
	state = deepcopy(state_0)
	for t in 1:max_t
		thetax_new = state.x - env.l*sin(state.θ)
		thetay_new = env.l*cos(state.θ)
		push!(xpositions,[state.x,thetax_new])
		push!(ypositions,[0,thetay_new])
		push!(xposcar,[state.x])
		push!(thetas,state.θ)
		push!(ws,state.w)
		push!(vs,state.v)
		push!(us,state.u)
		#actions_at_s,_ = adm_actions_b(state,env)
		# policy = ones(length(actions))./length(actions)
		ids_dstate,discretized_state = discretize_state(state,env)
		id_dstate = build_index_b(ids_dstate,env)
		if interpolation == true
			push!(values,value(state.θ,state.w,state.u,state.v,state.x))
		else
			push!(values,value[id_dstate])
		end
		actions_at_s,policy,n_best_actions = optimal_policy_q(discretized_state,state,value,ϵ,env,interpolation)
		#Choosing action randomly from optimal action set according to policy
		#action = rand(actions)
		#ϵ-greedy: with prob ϵ choose a random action from action set
		# if rand() < ϵ
		# 	action = rand(actions_at_s)
		# end
		idx = rand(Categorical(policy))
		action = actions_at_s[idx]
		#There might be degeneracy in optimal action, so choose with (1-ϵ)
		prob_distribution = ones(n_best_actions).*((1-ϵ)/n_best_actions)
		#Then, choose with ϵ between available actions
		for i in 1:length(actions_at_s)
			push!(prob_distribution,ϵ/length(actions_at_s))
		end
		#Compute entropy of that distribution
		push!(entropies,entropy(prob_distribution))
		#idx = rand(Categorical(policy))
		#action = actions[idx]
		push!(a_s,action)
		r = reachable_rewards(state,action,env)[1]
		push!(rewards,r)

		#For discretized dynamics
		#_,state_p = transition_b(state,action,env) 
		#For real dynamics
		if state.u == 1
			break
		end
		state_p = real_transition(state,action,env)
		state = deepcopy(state_p)
	#end
	end
	xpositions,ypositions,xposcar,thetas,ws,us,vs,a_s,values,entropies,rewards
end

# ╔═╡ 85092f44-1db7-4e29-ab56-12ef1985d90e
q_value_int(0.0,0.4,2,0.0,0.0)

# ╔═╡ 4893bf14-446c-46a7-b695-53bac123ff99
md" ## Animation"

# ╔═╡ 90aff8bb-ed69-40d3-a22f-112f713f4b93
md"# Comparison between H and Q agents"

# ╔═╡ 4fad0692-05dc-4c3b-9aae-cd9a43519e51
md"## ϵ-greedy policy survival rate analysis"

# ╔═╡ 973b56e5-ef3c-4328-ad26-3ab63650537e
ϵs_totry = [0.0,0.05,0.1,0.15,0.2]

# ╔═╡ 7edf8ddf-2b6e-4a4b-8181-6b8bbdd22841
begin
	max_t_q_anim = 1000
	state0_q_anim = State(θ = 0,x = 0, v = 0,w = 0, u = 2) #state_0_anim
	ϵ_anim = ϵs_totry[2]
	xs_q_anim, ys_q_anim, xposcar_q_anim, thetas_ep_q_anim, ws_ep_q_anim, us_ep_q_anim, vs_ep_q_anim, actions_q_anim, values_q_anim, entropies_q_anim,rewards_q_anim = create_episode_q(state0_q_anim,q_value_int,ϵ_anim,max_t_q_anim, ip_q, interpolation)
	length(xposcar_q_anim)
end

# ╔═╡ aa4229d3-e221-4a87-8a18-ed376d33d3ac
draw_cartpole_time(xposcar_q_anim,xs_q_anim,ys_q_anim,ip_b.max_x,200,1,ip_b)

# ╔═╡ 69128c7a-ddcb-4535-98af-24fc91ec0b7e
begin
	draw_cartpole_timeright(xposcar_q_anim,xs_q_anim,ys_q_anim,1000,1,ip_b)
	#savefig("q_frozen_movie2.pdf")
end

# ╔═╡ c98025f8-f942-498b-9368-5d524b141c62
if movies == true
 	anim_q = animate_w_borders(xposcar_q_anim,xs_q_anim,ys_q_anim,ip_q.max_x,max_t_q_anim,values_q_anim,entropies_q_anim,ip_q,true)
end

# ╔═╡ d8642b83-e824-429e-ac3e-70e875a47d1a
if movies == true
gif(anim_q,fps = Int(round(1/ip_q.Δt)),"episode_q_agent_epsilon_$(ϵ_anim)_g_$(ip_q.γ)_wvalues.gif")
end

# ╔═╡ 3d567640-78d6-4f76-b13e-95be6d5e9c64
begin
	#Calculate q values for several ϵs
	ϵs_to_compute_value = [0.25,0.3,0.35,0.4]
	for (i,ϵ) in enumerate(ϵs_to_compute_value)
		println("epsilon = ", ϵ)
		q_val, _, _ = Q_iteration(ip_q,ϵ,tol,1200,δ = 1E-5)
		writedlm("q_value_eps_$(ϵ)_nstates_$(ip_q.nstates).dat",q_val)
	end
end

# ╔═╡ 06f064cf-fc0d-4f65-bd6b-ddb6e4154f6c
begin
	max_time = 10000
	num_episodes = 1000
	ϵs = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4] #ϵs_totry# [0.0,0.001,0.01,0.05] 
	#To compute the survival times for various epsilon-greedy Q agents, it takes a long time
	# survival_pcts = zeros(length(ϵs),num_episodes)
	# for (i,ϵ) in enumerate(ϵs)
	# 	q_val = readdlm("q_value_eps_$(ϵ)_nstates_$(ip_q.nstates).dat")
	# 	q_val_int = interpolate_value(q_val,ip_q)
	# 	Threads.@threads for j in 1:num_episodes
	# 		println("j = ", j)
	# 		state0 = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval), u = 2)
	# 		xs_q, ys_q, xposcar_q, thetas_ep_q, ws_ep_q, us_ep_q, vs_ep_q, actions_q = create_episode_q(state0, q_val_int, ϵ, max_time, ip_q, interpolation)
	# 		#survival_timesq[j] = length(xposcar_q)
	# 		#if length(xposcar_q) == max_time
	# 			survival_pcts[i,j] = length(xposcar_q)
	# 		#end
	# 	end
	# end
	#writedlm("survival_pcts_Q_epsilon_short.dat",survival_pcts)
	
	#Computes it for H agent
	# survival_times = zeros(num_episodes)
	# Threads.@threads for i in 1:num_episodes
	# 	state0_b = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval), u = 2)
	# 	xs_b, ys_b, xposcar_b, thetas_ep_b, ws_ep_b, us_ep_b, vs_ep_b, actions_b = create_episode_b(state0_b,h_value_int,max_time, ip_b)
	# 	survival_times[i] = length(xposcar_b)
	# end

	#Otherwise, read from file
	survival_pcts = readdlm("survival_pcts_Q_epsilon_short.dat")
	survival_H = readdlm("survival_pcts_H.dat")
end;

# ╔═╡ e2f1952d-8fc8-4749-9a88-bf5c61758e14
#writedlm("survival_pcts_Q_epsilon_short.dat",survival_pcts)

# ╔═╡ 2bd1d018-986b-4370-95ba-e047e2c7b691
# begin
# 		#Computes it for H agent
# 		survival_times_H = zeros(num_episodes)
# 		Threads.@threads for i in 1:num_episodes
# 			state0_b = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval), u = 2)
# 			xs_b, ys_b, xposcar_b, thetas_ep_b, ws_ep_b, us_ep_b, vs_ep_b, actions_b = create_episode_b(state0_b,h_value_int,max_time, ip_b)
# 			survival_times_H[i] = length(xposcar_b)
# 		end
# end

# ╔═╡ 489c807d-2db1-4d9e-9e30-cab403b9281f
#writedlm("survival_pcts_H.dat",survival_times_H)

# ╔═╡ ac43808a-ef2a-472d-9a9e-c77257aaa535
survival_Q = survival_pcts

# ╔═╡ da37a5c8-b8ef-4131-a153-50c21461d9c4
# begin
# 	survival_Q_extra = zeros(5,10000)
# 	survival_Q_extra[1:3,:] = survival_Q[1:3,:]
# 	survival_Q_extra[4,:] = survival_pcts
# 	survival_Q_extra[5,:] = survival_Q[4,:]
# end

# ╔═╡ a8b1a61c-6bd0-417b-8d58-8c8f8d77da7e
#writedlm("survival_pcts_Q_agent.dat",survival_Q_extra)

# ╔═╡ 1a6c956e-38cc-4217-aff3-f303af0282a4
md"Density plot? $(@bind density CheckBox(default = false))"

# ╔═╡ e169a987-849f-4cc2-96bc-39f234742d93
begin
	bd = 2000
	surv_hists = plot(xlabel = "Survived time steps", xticks = [10000,50000,100000])
	if density == true
		plot!(surv_hists,ylabel = "Density")
		density!(surv_hists, bandwidth = bd, survival__H,label = "H agent",linewidth = 2)
	else
		plot!(surv_hists,ylabel = "Normalized frequency")
		plot!(surv_hists,bins = collect(-bd/2:bd:max_time+bd/2),survival__H,st = :stephist, label = "H agent", alpha = 1.0,linewidth = 2,normalized = :probability)
	end
	alphas = ones(length(ϵs))
	for i in 1:length(ϵs)
		if density == true
			density!(surv_hists, bandwidth = bd, survival_Q[i,:],label = "ϵ = $(ϵs[i])",linewidth = 2)
		else
			plot!(surv_hists,bins = (collect(-bd/2:bd:max_time+bd/2)),survival_Q[i,:],st = :stephist,normalized = :probability,label = "ϵ = $(ϵs[i])",alpha = alphas[i],linewidth = 2)
		end
	end
	plot(surv_hists, grid = true, minorgrid = false, legend_position = :top,margin = 4Plots.mm,size = (500,500))
	#savefig("q_h_survival_histograms.pdf")
end

# ╔═╡ ae15e079-231e-4ea2-aabb-ee7d44266c6d
begin
	surv_means = plot(xlabel = "ϵ")
	plot!(surv_means,ylabel = "Percentage of survived time steps")
	plot!(surv_means, ϵs, mean(survival_H./max_time).*ones(length(ϵs)),label = "H agent",linewidth = 2.5, ls = :dash)
	plot!(surv_means,ϵs,mean(survival_Q./max_time,dims = 2),yerror = std(survival_Q./max_time,dims = 2)./(sqrt(length(survival_Q[1,:]))),markerstrokewidth = 2, linewidth = 2.5,label = "R agent")
	plot(surv_means, grid = true, minorgrid = true, legend_position = :best,margin = 2Plots.mm,size = (400,550),bg_legend = :white,fg_legend = :white, fgborder = :red,fgguide = :black)
	#savefig("q_h_survival_epsilon_greedy.pdf")
end

# ╔═╡ 607b3245-53ab-4dca-9c98-8f4b179051e4
mean(survival_Q,dims = 2)

# ╔═╡ 2b5277cc-011f-4f3f-b860-2bf1dd568d21
length(survival_Q[1,:])

# ╔═╡ a4b26f44-319d-4b90-8fee-a3ab2418dc47
md"## State occupancy histograms"

# ╔═╡ 94da3cc0-6763-40b9-8773-f2a1a2cbe507
state_0_comp = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval),u=2)

# ╔═╡ 69b6d1f1-c46c-43fa-a21d-b1f61fcd7c55
time_histograms = 30000

# ╔═╡ 7a27d480-a9cb-4c26-91cd-bf519e8b35fa
begin
	max_t_b = time_histograms
	state0_b = state_0_comp
	xs_b, ys_b, xposcar_b, thetas_ep_b, ws_ep_b, us_ep_b, vs_ep_b, actions_b, values_b, entropies_b = create_episode_b(state0_b,h_value_int,max_t_b, ip_b)
	#Check if it survived the whole episode
	length(xposcar_b)
end

# ╔═╡ b367ccc6-934f-4b18-b1db-05286111958f
begin
	max_t_q = time_histograms
	state0_q = state_0_comp
	ϵ = 0.0
	q_value_hist = readdlm("q_value_eps_$(ϵ)_nstates_$(ip_q.nstates).dat")
	q_value_int_hist = interpolate_value(q_value_hist,ip_q)
	xs_q, ys_q, xposcar_q, thetas_ep_q, ws_ep_q, us_ep_q, vs_ep_q, actions_q, values_q, entropies_q,rewards_q = create_episode_q(state0_q,q_value_int_hist,ϵ,max_t_q, ip_q, interpolation)
	#Check if it survived the whole episode
	length(xposcar_q)
end

# ╔═╡ 096a58e3-f417-446e-83f0-84a333880680
begin
	x_b = Any[]
	x_q = Any[]
	for i in 1:length(xposcar_b)
	 push!(x_b,xposcar_b[i][1])
	end
	for i in 1:length(xposcar_q)
	 push!(x_q,xposcar_q[i][1])
	end
end

# ╔═╡ c95dc4f2-1f54-4266-bf23-7d24cee7b3d4
begin
	cbarlim = 0.006
	p1h = plot(ylim = (-36,36),xlim=(-1.5,1.5),xlabel = "Position", ylabel = "Angle",title = "H agent")
	#heatmap!(p1h,zeros(160,160))
	plot!(p1h,x_b,thetas_ep_b.*180/pi,bins = (40,40),st = :histogram2d,normed = :probability,clim = (0,cbarlim),cbar = false)
	p2h = plot(ylim = (-36,36),xlim=(-1.5,1.5),xlabel = "Position", title = "R agent")
	plot!(p2h,x_q,thetas_ep_q.*180/pi,bins = (40,40),st = :histogram2d,normed = :probability,clim = (0,cbarlim))
	plot(p1h,p2h,size = (900,400),layout = Plots.grid(1, 2, widths=[0.44,0.56]),margin = 5Plots.mm)
	#savefig("angle_position_histogram_eps_$(ϵ).png")
end

# ╔═╡ d2139819-83a1-4e2b-a605-91d1e686b9a3
# begin
# 	cbarlim_v = 0.005
# 	p1h_v = plot(xlim = (-ip_b.max_w*180/pi,ip_b.max_w*180/pi),ylim=(-ip_b.max_v,ip_b.max_v),xlabel = "Angular speed", ylabel = "Speed")
# 	#heatmap!(p1h,zeros(160,160))
# 	plot!(p1h_v,ws_ep_b.*180/pi,vs_ep_b,bins = (40,40),st = :histogram2d,normed = :probability,margin = 5Plots.mm,clim = (0,cbarlim_v),cbar = false)
# 	p2h_v = plot(xlim = (-ip_b.max_w*180/pi,ip_b.max_w*180/pi),ylim=(-ip_b.max_v,ip_b.max_v),xlabel = "Angular speed")
# 	plot!(p2h_v,ws_ep_q.*180/pi,vs_ep_q,bins = (40,40),st = :histogram2d,normed = true,margin = 5Plots.mm,clim = (0,cbarlim_v))
# 	plot(p1h_v,p2h_v,size = (1000,500))
# end

# ╔═╡ fa9ae665-465d-4f3e-b8cd-002c80420adb
# begin
# 	cbarlim2 = 0.003
# 	p1h_p = plot(xlim = (-36,36),ylim=(-ip_b.max_w*180/pi,ip_b.max_w*180/pi),xlabel = "Angle", ylabel = "Angular speed")
# 	#heatmap!(p1h,zeros(160,160))
# 	plot!(p1h_p,thetas_ep_b.*180/pi,ws_ep_b.*180/pi,bins = (40,40),st = :histogram2d,normed = :probability,margin = 5Plots.mm,clim = (0,cbarlim2),cbar = false)
# 	p2h_p = plot(xlim = (-36,36),ylim=(-ip_b.max_w*180/pi,ip_b.max_w*180/pi),xlabel = "Angle")
# 	plot!(p2h_p,thetas_ep_q.*180/pi,ws_ep_q.*180/pi,bins = (40,40),st = :histogram2d,normed = :probability,margin = 5Plots.mm,clim = (0,cbarlim2))
# 	plot(p1h_p,p2h_p,size = (1000,500))
# end

# ╔═╡ 6cc17343-c725-4653-ad48-b3535a53b09e
# begin
# 	cbarlim3 = 0.003
# 	p1h_p2 = plot(xlim = (-1.5,1.5),ylim=(-ip_b.max_v,ip_b.max_v),xlabel = "Position", ylabel = "Speed")#, title = "H agent")
# 	#heatmap!(p1h,zeros(160,160))
# 	plot!(p1h_p2,x_b,vs_ep_b,bins = (40,40),st = :histogram2d,normed = :probability,clim = (0,cbarlim3),cbar = false)
# 	p2h_p2 = plot(xlim = (-1.5,1.5),ylim=(-ip_b.max_v,ip_b.max_v),xlabel = "Position")#, title = "R agent")
# 	plot!(p2h_p2,x_q,vs_ep_q,bins = (40,40),st = :histogram2d,normed = :probability,clim = (0,cbarlim3))
# 	plot(p1h_p2,p2h_p2,size = (900,400),layout = Plots.grid(1, 2, widths=[0.45,0.55]),margin = 7Plots.mm)
# 	#savefig("speed_position_histogram.svg")
# end

# ╔═╡ 9fa87325-4af3-4ec0-a716-80652dcf2ace
md"Angle vs Position? $(@bind angle CheckBox(default = false))"

# ╔═╡ c9e4bc33-046f-4ebd-9da7-a768886107ef
if angle
	limy = ip_b.max_θ*180/pi
	labely = "Angle"
else
	limy = ip_b.max_v
	labely = "Speed"
end

# ╔═╡ f4b1a0bf-aff3-4b49-b15f-e5fdf31d969c
begin
	phase_space_trajectory = plot(1,xlabel = "Position",ylabel = labely,xlim = (-1.5,1.5),ylim=(-limy,limy),label = false, title = "R agent",marker = (:black,2))
	@gif for i in 1:200#length(x_b)
		if angle
			push!(phase_space_trajectory,x_q[i],thetas_ep_q[i]*180/pi)
		else
			push!(phase_space_trajectory,x_q[i],vs_ep_q[i])
		end
	end every 10
end

# ╔═╡ bde8e418-9391-4c5a-a754-5ae1e3215e66
begin
	ps_trajectory = plot(1,xlabel = "Position",ylabel = labely,xlim = (-1.5,1.5),ylim=(-limy,limy),label = false, title = "H agent",marker = (:black,2))
	@gif for i in 1:200#length(x_b)
		if angle
			push!(ps_trajectory,x_b[i],thetas_ep_b[i]*180/pi)
		else
			push!(ps_trajectory,x_b[i],vs_ep_b[i])
		end
	end every 10
end

# ╔═╡ d0c487cb-041f-4c8d-9054-e5f3cfad1ed4
begin
	#title = plot(title = "Initial state: θ = $(round(state_0_comp.θ*180/pi,sigdigits=2)) deg, ω = $(round(state_0_comp.w*180/pi,sigdigits=2)) deg/s, x = $(round(state_0_comp.x,sigdigits=2)) m,  v = $(round(state_0_comp.v,sigdigits=2)) m/s", grid = false, axis = false, ticks = false, bottom_margin = -20Plots.px)
	p1 = plot(xlim=(-36,36), xlabel = "Angle (deg)")
	plot!(p1,thetas_ep_b.*180/pi,st = :density,label = false,lw = 2)
	plot!(p1,thetas_ep_q.*180/pi,st = :density,label = false,lw = 2)
	plot!(p1,ylabel = "Density")
	p2 = plot(xlim=(-3.5*180/pi,3.5*180/pi), xlabel = "Angular speed (deg/s)")
	plot!(p2,ws_ep_b.*180/pi,st = :density,label = "h agent",lw = 2)
	plot!(p2,ws_ep_q.*180/pi,st = :density,label = "q agent, \nϵ = $(ϵ)",lw = 2)
	plot!(p2, legend_foreground_color = nothing,legend_position = :topright)
	p3 = plot(xlim=(-2.,2.))
	#plot!(p3,bins = collect(-2.05:0.1:2.05),x_b,st = :stephist)
	#plot!(p3,bins = collect(-2.05:0.1:2.05),x_q,st = :stephist)
	plot!(p3,x_b,st = :density, label = false, xlabel = "Position",lw = 2)
	plot!(p3,x_q,st = :density,label = false,lw = 2)
	plot!(p3,ylabel = "Density")
	p4 = plot(xlim=(-4,4))
	plot!(p4,vs_ep_b,st = :density, bandwidth = 0.1,label = false, xlabel = "Linear speed",lw = 2)
	plot!(p4,vs_ep_q,st = :density, bandwidth = 0.1,label = false,lw = 2)
	plot!(p4)
	#plot(title,p1,p2,p3,p4, size = (900,500), layout = @layout([A{0.01h}; [B C]; [D E]]))
	plot(p1,p2,p3,p4, size = (900,600), margin = 4Plots.mm)
	#savefig("histograms_epsilon_$(ϵ).pdf")
end

# ╔═╡ c283afa6-b3cf-4161-b949-732aa2464eb7
md"## Action histogram"

# ╔═╡ c982e800-6089-4860-a190-47f66f802d6d
begin
	plot(actions_b, bins = [ip_b.a_s[1]-1,ip_b.a_s[1]+1,ip_b.a_s[2]-1,ip_b.a_s[2]+1,ip_b.a_s[3]-1,ip_b.a_s[3]+1,ip_b.a_s[4]-1,ip_b.a_s[4]+1,ip_b.a_s[5]-1,ip_b.a_s[5]+1], st = :stephist, label = "h agent")
	plot!(actions_q, bins = [ip_b.a_s[1]-1,ip_b.a_s[1]+1,ip_b.a_s[2]-1,ip_b.a_s[2]+1,ip_b.a_s[3]-1,ip_b.a_s[3]+1,ip_b.a_s[4]-1,ip_b.a_s[4]+1,ip_b.a_s[5]-1,ip_b.a_s[5]+1], st = :stephist, label = "q agent")
#savefig("actions_smallreward_agents_nstates_$(ip_q.nstates)_ep$(Int(max_t_q*ip_q.Δt)).pdf")
end

# ╔═╡ Cell order:
# ╠═63a9f3aa-31a8-11ec-3238-4f818ccf6b6c
# ╠═11f37e01-7958-4e79-a8bb-06b92d7cb9ed
# ╠═0a29d50e-df0a-4f54-b588-faa6aee2e983
# ╠═18382e9c-e812-49ff-84cc-faad709bc4c3
# ╠═379318a2-ea2a-4ac1-9046-0fdfe8c102d4
# ╟─40ee8d98-cc45-401e-bd9d-f9002bc4beba
# ╠═4ec0dd6d-9d3d-4d30-8a19-bc91600d9ec2
# ╟─f997fa16-69e9-4df8-bed9-7067e1a5537d
# ╟─2f31b286-11aa-443e-a3dc-c021e6fc276c
# ╟─4263babb-32ae-446f-b6a6-9d5451ed40cd
# ╟─b1d3fff1-d980-4cc8-99c3-d3db7a71bf60
# ╟─9396a0d1-6036-44ec-98b7-16df4d150b54
# ╟─cfdf3a8e-a377-43ef-8a76-63cf78ce6a16
# ╟─ffe32878-8732-46d5-b57c-9e9bb8e6dd74
# ╟─784178c3-4afc-4c65-93e1-4265e1faf817
# ╟─fa56d939-09d5-4b63-965a-38016c957fbb
# ╟─05b8a93c-cefe-473a-9f8e-3e82de0861b2
# ╟─f92e8c6d-bf65-4786-8782-f38847a7fb7a
# ╠═f9121267-8d7e-40b1-b9cc-c8da3f45cdb8
# ╟─36601cad-1ba9-48f2-8463-a58f98bedd34
# ╠═ce8bf897-f240-44d8-ae39-cf24eb115704
# ╟─b975eaf8-6a94-4e39-983f-b0fb58dd70a1
# ╠═e0fdce26-41b9-448a-a86a-6b29b68a6782
# ╠═bbe19356-e00b-4d90-b704-db33e0b75743
# ╠═b38cdb2b-f570-41f4-8996-c7e41551f374
# ╠═dafbc66b-cbc2-41c9-9a42-da5961d2eaa6
# ╠═2735cbf5-790d-40b4-ac32-a413bc1d530a
# ╟─79372251-157d-43a0-9560-4727fbd36ea9
# ╠═2f1b6992-3082-412d-b966-2b4b278b2ed0
# ╠═aa4229d3-e221-4a87-8a18-ed376d33d3ac
# ╠═10a47fa8-f235-4455-892f-9b1457b1f82c
# ╠═a54788b9-4b1e-4066-b963-d04008bcc242
# ╠═69128c7a-ddcb-4535-98af-24fc91ec0b7e
# ╟─e099528b-37a4-41a2-836b-69cb3ceda2f5
# ╠═6b9b1c38-0ed2-4fe3-9326-4670a33e7765
# ╠═7c04e2c3-4157-446d-a065-4bfe7d1931fd
# ╠═a0b85a14-67af-42d6-b231-1c7d0c293f6e
# ╠═8a59e209-9bb6-4066-b6ca-70dac7da33c3
# ╠═08ecfbed-1a7c-43d4-ade7-bf644eeb6eee
# ╟─21472cb5-d968-4306-b65b-1b25f522dd4d
# ╠═c5c9bc66-f554-4fa8-a9f3-896875a50627
# ╠═370a2da6-6de0-44c0-9f70-4e676769f59b
# ╠═9230de54-3ee3-4242-bc34-25a38edfbb6b
# ╟─24a4ba06-d3ae-4c4b-9ab3-3852273c2fd4
# ╟─63fe2f55-f5ce-4022-99ee-3bd4e14e6352
# ╠═14314bcc-661a-4e84-886d-20c89c07a28e
# ╠═c087392c-c411-4368-afcc-f9a104856884
# ╟─6107a0ce-6f01-4d0b-bd43-78f2125ac185
# ╠═e181540d-4334-47c4-b35d-99023c89a2c8
# ╠═31a1cdc7-2491-42c1-9988-63650dfaa3e3
# ╟─c05d12f3-8b8a-4f34-bebc-77e376a980d0
# ╠═87c0c3cb-7059-42ae-aed8-98a0ef2eb55f
# ╟─f9b03b4d-c521-4456-b0b9-d4a301d8a813
# ╠═355db008-9661-4e54-acd5-7c2c9ba3c7f5
# ╠═564cbc7a-3125-4b67-843c-f4c74ccef51f
# ╠═d20c1afe-6d5b-49bf-a0f2-a1bbb21c709f
# ╠═e9687e3f-be56-44eb-af4d-f169558de0fd
# ╠═e9a01524-90b1-4249-a51c-ec8d1624be5b
# ╠═85092f44-1db7-4e29-ab56-12ef1985d90e
# ╟─4893bf14-446c-46a7-b695-53bac123ff99
# ╠═7edf8ddf-2b6e-4a4b-8181-6b8bbdd22841
# ╠═c98025f8-f942-498b-9368-5d524b141c62
# ╠═d8642b83-e824-429e-ac3e-70e875a47d1a
# ╟─90aff8bb-ed69-40d3-a22f-112f713f4b93
# ╟─4fad0692-05dc-4c3b-9aae-cd9a43519e51
# ╠═973b56e5-ef3c-4328-ad26-3ab63650537e
# ╠═3d567640-78d6-4f76-b13e-95be6d5e9c64
# ╠═06f064cf-fc0d-4f65-bd6b-ddb6e4154f6c
# ╠═e2f1952d-8fc8-4749-9a88-bf5c61758e14
# ╠═2bd1d018-986b-4370-95ba-e047e2c7b691
# ╠═489c807d-2db1-4d9e-9e30-cab403b9281f
# ╠═ac43808a-ef2a-472d-9a9e-c77257aaa535
# ╠═da37a5c8-b8ef-4131-a153-50c21461d9c4
# ╠═a8b1a61c-6bd0-417b-8d58-8c8f8d77da7e
# ╟─1a6c956e-38cc-4217-aff3-f303af0282a4
# ╠═e169a987-849f-4cc2-96bc-39f234742d93
# ╠═ae15e079-231e-4ea2-aabb-ee7d44266c6d
# ╠═607b3245-53ab-4dca-9c98-8f4b179051e4
# ╠═2b5277cc-011f-4f3f-b860-2bf1dd568d21
# ╟─a4b26f44-319d-4b90-8fee-a3ab2418dc47
# ╠═94da3cc0-6763-40b9-8773-f2a1a2cbe507
# ╠═69b6d1f1-c46c-43fa-a21d-b1f61fcd7c55
# ╠═7a27d480-a9cb-4c26-91cd-bf519e8b35fa
# ╠═b367ccc6-934f-4b18-b1db-05286111958f
# ╠═096a58e3-f417-446e-83f0-84a333880680
# ╠═c95dc4f2-1f54-4266-bf23-7d24cee7b3d4
# ╠═d2139819-83a1-4e2b-a605-91d1e686b9a3
# ╠═fa9ae665-465d-4f3e-b8cd-002c80420adb
# ╠═6cc17343-c725-4653-ad48-b3535a53b09e
# ╟─9fa87325-4af3-4ec0-a716-80652dcf2ace
# ╠═c9e4bc33-046f-4ebd-9da7-a768886107ef
# ╠═f4b1a0bf-aff3-4b49-b15f-e5fdf31d969c
# ╠═bde8e418-9391-4c5a-a754-5ae1e3215e66
# ╠═d0c487cb-041f-4c8d-9054-e5f3cfad1ed4
# ╟─c283afa6-b3cf-4161-b949-732aa2464eb7
# ╠═c982e800-6089-4860-a190-47f66f802d6d
