### A Pluto.jl notebook ###
# v0.17.1

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

# ╔═╡ a649da15-0815-438d-9bef-02c6d204656e
begin
	using Pkg
	Pkg.activate("../Project.toml")
	using Plots, Distributions, PlutoUI, Parameters
	using StatsPlots, DelimitedFiles, ZipFile
end

# ╔═╡ 25c56490-3b9c-4825-b91d-8b9e41fc0f6b
using LinearAlgebra

# ╔═╡ 422493c5-8a90-4e70-bd06-40f8e6b254f1
gr()

# ╔═╡ 76f77726-7776-4975-9f30-3887f13ae3e7
default(titlefont = ("Computer Modern",16), legend_font_family = "Computer Modern", legend_font_pointsize = 14, guidefont = ("Computer Modern", 16), tickfont = ("Computer Modern", 14))

# ╔═╡ 393eaf2d-e8fe-4675-a7e6-32d0fe9ac4e7
begin
	PlutoUI.TableOfContents(aside = true)
end

# ╔═╡ b4e7b585-261c-4044-87cc-cbf669768145
#Time steps for animations
max_t_anim = 200

# ╔═╡ 7feeec1a-7d7b-4220-917d-049f1e9b101b
md"# Grid world environment"

# ╔═╡ 7e68e560-45d8-4429-8bff-3a8229c8c84e
@with_kw struct environment
	γ::Float64 = 0.96
	sizex::Int64 = 5
	sizey::Int64 = 5
	sizeu::Int64 = 10
	xborders::Vector{Int64} = [0,sizex+1]
	yborders::Vector{Int64} = [0,sizey+1]
	#Location of obstacles
	obstacles
	obstaclesx
	obstaclesy
	#Array of vectors where rewards can be found
	reward_locations
	reward_mags
end

# ╔═╡ 194e91cb-b619-4908-aebd-3136107175b7
function adm_actions(s_state,u_state,env::environment, constant_actions = false)
	out = Any[]
	moving_actions = [[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]]
	ids_actions = collect(1:length(moving_actions))
	#If agent is "alive"
	if u_state > 1
		#To check that agent does not start at harmful state
		#if transition_u(s_state,u_state,[0,0],env) != 1
			out = deepcopy(moving_actions)
			#When we constrain by hand the amount of actions
			#we delete some possible actions
			if constant_actions == false
				#Give all possible actions by default
				#Check the boundaries of gridworld
				for it in 1:2
				not_admissible = in(env.xborders).(s_state[1]+(-1)^it)
					if not_admissible == true
						ids = findall(idx -> idx[1] == (-1)^it,out)
						deleteat!(out,ids)
						deleteat!(ids_actions,ids)
					end
				end
				for it in 1:2
				not_admissible = in(env.yborders).(s_state[2]+(-1)^it)
					if not_admissible == true
						ids = findall(idx -> idx[2] == (-1)^it,out)
						deleteat!(out,ids)
						deleteat!(ids_actions,ids)
					end
				end
				#Check for obstacles
				for action in moving_actions
					idx = findall(i -> i == s_state+action,env.obstacles)
					if length(idx) > 0
						idx2 = findfirst(y -> y == action,out)
						deleteat!(out,idx2)
						deleteat!(ids_actions,idx2)
					end
				end
			#end
		end
	else
		ids_actions = Any[]
	end
	#Doing nothing is always an admissible action
	push!(out,[0,0])
	push!(ids_actions,length(ids_actions)+1)
	out,ids_actions
	#Checking if having all actions at every state changes results
	#return [[1,0],[0,1],[-1,0],[0,-1],[0,0]],[1,2,3,4,5]
end

# ╔═╡ a46ced5b-2e58-40b2-8eb6-b4840043c055
md"## Functions for dynamic programming"

# ╔═╡ 9404080e-a52c-42f7-9abd-ea488bf7abc2
function reachable_states(s_state,a)
	#Deterministic environment
	s_prime = s_state + a
	[s_prime]
end

# ╔═╡ 8675158f-97fb-4222-a32b-49ce4f6f1d41
function transition_s(s,a,env)
	s_prime = s + a
	if in(env.obstacles).([s_prime]) == [true]
		s_prime = s
	else
		if s_prime[1] == env.xborders[1] || s_prime[1] == env.xborders[2] || s_prime[2] == env.yborders[1] || s_prime[2] == env.yborders[2]
			s_prime = s
		end
	end
	s_prime
end

# ╔═╡ 0dcdad0b-7acc-4fc4-93aa-f6eacc077cd3
function rewards(s,a,env::environment)
	rewards = 0
	s_p = transition_s(s,a,env)
	for i in 1:length(env.reward_locations)
		if s == env.reward_locations[i]
			rewards += env.reward_mags[i]
		end
	end
	action_cost = 0
	#Only deduct for action if action had an impact on the world
	action_cost = (abs(a[1]) + abs(a[2]))
	#locs = findall(i -> equal(env.reward_locations[i], s_p))
	rewards - action_cost - 1
end

# ╔═╡ 0ce119b1-e269-41e2-80b7-991cae37cf5f
function transition_u(s,u,a,env)
	u_prime = u + rewards(s,a,env)
	if u_prime > env.sizeu
		u_prime = env.sizeu
	elseif u_prime <= 0
		u_prime = 1
	elseif u == 1
		u_prime = 1
	end
	u_prime
end

# ╔═╡ 92bca36b-1dc9-4c03-88c0-6a684dfbec9f
md"## Helper functions"

# ╔═╡ c96e3331-1dcd-4b9c-b28d-d74493c8934d
function build_index(s::Vector{Int64},u::Int64,env::environment)
	Int64(s[1] + env.sizex*(s[2]-1) + env.sizex*env.sizey*(u-1))
end

# ╔═╡ d0a5c0fe-895f-42d8-9db6-3b0fcc6bb43e
md" # Environment"

# ╔═╡ 6c716ad4-23c4-46f8-ba77-340029fcce87
function initialize_env(γ, size_x,size_y,capacity,which,obstacles,reward_locations,reward_mags) #
	if which == "wall"
		wall_x_location = [2]
	else
		wall_x_location = [size_x + 2]
	end
	if which == "wall"
		for idx in 1:length(wall_x_location)
			for y in 1:size_y
				if y != 1 #Int(floor((sizey+1)/2))
					push!(obstacles,[wall_x_location[idx],y])
				end
			end
		end
	end
	obstaclesx = Any[]
	obstaclesy = Any[]
	for i in 1:length(obstacles)
		push!(obstaclesx,obstacles[i][1])
		push!(obstaclesy,obstacles[i][2])
	end
environment(γ = γ, sizex =
 size_x,sizey = size_y,sizeu = capacity,obstacles = obstacles,obstaclesx = obstaclesx,obstaclesy = obstaclesy,reward_locations = reward_locations,reward_mags = reward_mags)
end

# ╔═╡ 07abd5b7-b465-425b-9823-19b73d07db56
@bind which PlutoUI.Select(["free" => "Free environment","wall" => "Environment with wall"], default = "free")

# ╔═╡ 8f2fdc23-1b82-4479-afe7-8eaf3304a122
begin
	γ = 0.96
	tol = 1E-3
	n_iter = 2000
	size_x = 10
	size_y = 10
	capacity = 50
	reward_locations = [[1,1],[size_x,size_y]]
	reward_mags = [5,50]
	obstacles = [[1,4],[2,4],[3,4],[4,3],[4,2],[4,1]]
	#Rewards for the walled environment
	#reward_locations = [[1,6],[3,2],[4,2],[5,2],[6,2]]
	#reward_mags = [30,-40,-30,-20,-10]
end

# ╔═╡ fa33c170-e46e-444b-92de-cd80327a7b0a
env1 = initialize_env(γ,size_x,size_y,capacity,which,obstacles,reward_locations,reward_mags);

# ╔═╡ 5e6964a3-7169-4c0d-b6d3-3f11d3cc7d36
env2 = initialize_env(γ,size_x,size_y,200,which,obstacles,reward_locations,reward_mags)

# ╔═╡ 403a06a7-e30f-4aa4-ade1-55dee37cd514
function draw_environment(x_pos,y_pos,u,env::environment)
	reward_sizes = [10,20]
	#Draw obstacles
	ptest = scatter(env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 20, color = "black")
	#Draw food
	for i in 1:length(reward_mags)
		scatter!(ptest,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i],color = "green",markershape = :diamond)
	end
	#Draw agent
	plot!(ptest, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	#Draw arrows
	actions,_ = adm_actions([x_pos[1],y_pos[1]],u[1],env)
	arrow_x = zeros(length(actions))
	arrow_y = zeros(length(actions))
	aux = actions
	for i in 1:length(aux)
		mult = 1
		if norm(aux[i]) > 1
			mult = 1/sqrt(2)
		end
		arrow_x[i] = aux[i][1]*mult*1.3
		arrow_y[i] = aux[i][2]*mult*1.3
	end
	quiver!(ptest,ones(Int64,length(aux))*x_pos[1],ones(Int64,length(aux))*y_pos[1],quiver = (arrow_x,arrow_y),color = "black",linewidth = 3)
	
	#Draw internal states
	ptest2 = plot(xticks = false,ylim = (0,env.sizeu), grid = false,legend = false)
	scatter!(ptest, x_pos,y_pos, markersize = 20, leg = false, color = "red")
	bar!(ptest2, u, color = "green")
	plot(ptest,ptest2,layout = Plots.grid(1, 2, widths=[0.8,0.2]), title=["" "u(t)"], size = (700,500))
end

# ╔═╡ ac3a4aa3-1edf-467e-9a47-9f6d6655cd04
md"# H agent"

# ╔═╡ c6870051-0241-4cef-9e5b-bc876a3894fa
function h_iteration(env::environment,tolerance = 1E-2, n_iter = 100,verbose = false)
	value = zeros(env.sizex,env.sizey,env.sizeu)
	value_old = deepcopy(value)
	t_stop = n_iter
	f_error = 0
	for t in 1:n_iter
		ferror_max = 0
		Threads.@threads for u in 1:env.sizeu
			for x in 1:env.sizex, y in 1:env.sizey
				s = [x,y]
				ids = findall(i -> i == s,env.obstacles)
				#Only update value for non-obstacle states
				if length(ids) == 0
					actions,_ = adm_actions(s,u,env)
					Z = 0
					for a in actions
						s_primes = reachable_states(s,a)
						for s_prime in s_primes
							u_prime = transition_u(s,u,a,env)
							Z += exp(env.γ*value_old[s_prime[1],s_prime[2],u_prime])
						end
					end
					value[x,y,u] = log(Z)
					f_error = abs(value[x,y,u] - value_old[x,y,u])
					ferror_max = max(ferror_max,f_error)
				else
					value[x,y,u] = 0
				end
			end
		end
		if ferror_max < tolerance
			t_stop = t
			break
		end
		if verbose == true 
			println("iteration = ", t, ", max function error = ", ferror_max)
		end
		value_old = deepcopy(value)
	end
	value,t_stop
end

# ╔═╡ d88e0e27-2354-43ad-9c26-cdc90beeea0f
function optimal_policy(s,u,optimal_value,env::environment,verbose = false)
	actions,_ = adm_actions(s,u,env)
	policy = zeros(length(actions))
	Z = exp(optimal_value[s[1],s[2],u])
	#Only compute policy for available actions
	for (idx,a) in enumerate(actions)
		u_p = transition_u(s,u,a,env)
		s_p = transition_s(s,a,env)
		policy[idx] = exp(env.γ*optimal_value[s_p[1],s_p[2],u_p]-optimal_value[s[1],s[2],u])
	end
	#adjust for numerical errors in probability
	sum_p = sum(policy)
	if verbose == true
		println("state = ", s, " u = ", u)
		println("policy = ", policy)
		println("sum policy = ", sum(policy))
	end
	policy = policy./sum(policy)
	actions,policy
end
		

# ╔═╡ 184636e2-c87d-4a89-b231-ff4aef8424d5
md"## Optimal value function"

# ╔═╡ 82fbe5a0-34a5-44c7-bdcb-36d16f09ea7b
begin
	h_value,t_stop = h_iteration(env1,tol,n_iter,true)
	writedlm("h_value_u_$(env1.sizeu).dat",h_value)
end;

# ╔═╡ e67db730-ca7c-4ef4-a2d2-7e001d5f7a79
u = 20
#@bind u PlutoUI.Slider(1:env1.sizeu)

# ╔═╡ 73722c01-adee-4bfd-97b4-60f2ced23725
function plot_optimal_policy(p,u,opt_value,env::environment,constant_actions = false)
	for x in 1:env.sizex
		for y in 1:env.sizey 
			ids = findall(i -> i == [x,y],env.obstacles)
			if length(ids) == 0
				actions,probs = optimal_policy([x,y],u,opt_value,env,false)
				arrow_x = zeros(length(actions))
				arrow_y = zeros(length(actions))
				aux = actions.*probs
				for i in 1:length(aux)
					arrow_x[i] = aux[i][1]*1.5
					arrow_y[i] = aux[i][2]*1.5
				end
				quiver!(p,ones(Int64,length(aux))*x,ones(Int64,length(aux))*y,quiver = (arrow_x,arrow_y),color = "green",linewidth = 2)
				scatter!(p,ones(Int64,length(aux))*x + arrow_x, ones(Int64,length(aux))*y + arrow_y,markersize = probs*30, color = "red")
			end
		end
	end
end		

# ╔═╡ 76f506dc-b21d-4e13-a8e8-9d1b3bd21b30
begin
	p1 = heatmap(transpose(h_value[:,:,u]), title = "optimal value function, u = $u",clims = (minimum(h_value[1,:,u]),maximum(h_value[1,:,u])))
	for i in 1:length(env1.reward_mags)
		if reward_mags[i]  > 0
			col = "green"
		else
			col = "gray"
		end
		scatter!(p1,[env1.reward_locations[i][1]],[env1.reward_locations[i][2]], color = col, markersize = min(abs(env1.reward_mags[i]),50))
	end
	plot!(p1,size = (1000,800)) 
	plot_optimal_policy(p1,u,h_value,env1)
	#p2 = heatmap(val_stoch[:,:,u], title = "value function random walk")
	plot(p1,legend = false)
	#savefig("optimal_value_function.png")
end

# ╔═╡ aa5e5bf6-6504-4c01-bb36-df0d7306f9de
md"## Sample trajectory"

# ╔═╡ ef9e78e2-d61f-4940-9e62-40c6d060353b
function sample_trajectory(s_0,u_0,opt_value,max_t,env::environment)
	xpositions = Any[]
	ypositions = Any[]
	u_states = Any[]
	all_x = Any[]
	all_y = Any[]
	urgency = Any[]
	push!(xpositions,[s_0[1]])
	push!(ypositions,[s_0[2]])
	push!(u_states,u_0)
	s = s_0
	u = u_0
	for t in 1:max_t
		actions,policy = optimal_policy(s,u,opt_value,env)
		idx = rand(Categorical(policy))
		action = actions[idx]
		for i in 1:length(actions)
			if policy[i] > 0.6
				push!(urgency,u)
			end
		end
		s_p = transition_s(s,action,env)
		u_p = transition_u(s,u,action,env)
		s = s_p
		u = u_p
		push!(xpositions,[s[1]])
		push!(ypositions,[s[2]])
		push!(all_x,s[1])
		push!(all_y,s[2])
		push!(u_states,u)		
	end
	xpositions,ypositions,u_states,all_x,all_y,urgency
end

# ╔═╡ a4457d71-27dc-4c93-81ff-f21b2dfed41d
md"### A movie"

# ╔═╡ 7ad00e90-3431-4e61-9a7f-efbc14d0724e
function animation(x_pos,y_pos,us,max_t,env::environment)
anim = @animate for t in 1:max_t+2
	reward_sizes = [10,20]
	#Draw obstacles
	ptest = scatter(env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 20, color = "black")
	#Draw food
	for i in 1:length(reward_mags)
		scatter!(ptest,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i],color = "green",markershape = :diamond)
	end
	plot!(ptest, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	ptest2 = plot(xticks = false,ylim = (0,env.sizeu), grid = false,legend = false)
	if t <= max_t
		scatter!(ptest, x_pos[t],y_pos[t], markersize = 15, leg = false, color = "blue")
	bar!(ptest2, [us[t]])
		if us[t] == 1
			scatter!(ptest,x_pos[t],y_pos[t],markersize = 30,markershape = :xcross,color = "black")
		end
	else 
		scatter!(ptest,[(env.sizex+1)/2],[(env.sizey+1)/2],markersize = 100,markershape = :square, color = "gray",leg = false)
	end
	plot(ptest,ptest2,layout = Plots.grid(1, 2, widths=[0.8,0.2]), title=["" "u(t)"], size = (1000,600))
end
end

# ╔═╡ b072360a-6646-4d6d-90ea-716085c53f66
md"Produce animation? $(@bind movie CheckBox(default = false))"

# ╔═╡ f45ca37a-cba5-41e8-8058-1138e58daf73
md"#### Deterministic behavior as a function of internal states"

# ╔═╡ f98d6ea0-9d98-4940-907c-455397158f3b
md"# Q agent"

# ╔═╡ 5f4234f5-fc0e-4cdd-93ea-99b6463b2ba1
function reachable_rewards(s,u,a,env::environment)
	r = 1
	if u < 2
		r = 0
	end
	r
end

# ╔═╡ 7a0173ac-240d-4f93-b413-45c6af0f4011
function q_iteration(env::environment,tolerance = 1E-2, n_iter = 100,verbose = false)
	value = zeros(env.sizex,env.sizey,env.sizeu)
	value_old = deepcopy(value)
	t_stop = n_iter
	f_error = 0
	for t in 1:n_iter
		ferror_max = 0
		Threads.@threads for u in 1:env.sizeu
			for x in 1:env.sizex, y in 1:env.sizey
				s = [x,y]
				ids = findall(i -> i == s,env.obstacles)
				#Only update value for non-obstacle states
				if length(ids) == 0
					actions,_ = adm_actions(s,u,env)
					values = zeros(length(actions))
					for (id_a,a) in enumerate(actions)
						s_primes = reachable_states(s,a)
						r = reachable_rewards(s,u,a,env)
						for s_prime in s_primes
							u_prime = transition_u(s,u,a,env)
							values[id_a] += r + env.γ*value_old[s_prime[1],s_prime[2],u_prime]
						end
					end
					value[x,y,u] = maximum(values)
					f_error = abs(value[x,y,u] - value_old[x,y,u])
					ferror_max = max(ferror_max,f_error)
				else
					value[x,y,u] = 0
				end
			end
		end
		if ferror_max < tolerance
			t_stop = t
			break
		end
		if verbose == true 
			println("iteration = ", t, ", max function error = ", ferror_max)
		end
		value_old = deepcopy(value)
	end
	value,t_stop
end

# ╔═╡ caadeb3b-0938-4559-8122-348c960a6eb1
q_value,t_stop_q = q_iteration(env1,tol,n_iter,true);

# ╔═╡ 29a4d235-8b03-4701-af89-cd289f212e7d
writedlm("q_value_u_$(env1.sizeu).dat",q_value)

# ╔═╡ 819c1be2-339f-4c37-b8a3-9d8cb6be6496
u_q = 20

# ╔═╡ 358bc5ca-c1f6-40f1-ba2d-7e8466531903
begin
	p1_q = heatmap(transpose(q_value[:,:,u_q]), title = "optimal value function, u = $u_q",clims = (minimum(q_value[1,:,u_q]),maximum(q_value[1,:,u_q])))
	for i in 1:length(env1.reward_mags)
		if reward_mags[i]  > 0
			col = "green"
		else
			col = "gray"
		end
		scatter!(p1_q,[env1.reward_locations[i][1]],[env1.reward_locations[i][2]], color = col, markersize = min(abs(env1.reward_mags[i]),50))
	end
	plot!(p1_q,size = (1000,800)) 
	#plot_optimal_policy(p1,u,h_value,env1)
	#p2 = heatmap(val_stoch[:,:,u], title = "value function random walk")
	plot(p1_q,legend = false)
	#savefig("optimal_value_function.png")
end

# ╔═╡ 40d62df0-53bb-4b46-91b7-78ffd621a519
function optimal_policy_q(s,u,value,env::environment)
	actions,ids_actions = adm_actions(s,u,env)
	q_values = zeros(length(actions))
	for (idx,a) in enumerate(actions)
		s_primes = reachable_states(s,a)
		r = reachable_rewards(s,u,a,env)
		for s_p in s_primes
			#deterministic environment
			u_p = transition_u(s,u,a,env)
			q_values[idx] += r + env.γ*value[s_p[1],s_p[2],u_p]
		end
	end
	best_actions = findall(i-> i == maximum(q_values),q_values)
	actions[best_actions],length(best_actions)
end

# ╔═╡ 005720d3-5920-476b-9f96-39971f512452
optimal_policy_q([3,3],20,q_value,env1)

# ╔═╡ 2379dcc3-53cb-4fb6-b1e8-c851e36acd1f
function sample_trajectory_q(s_0,u_0,opt_value,max_t,env::environment)
	xpositions = Any[]
	ypositions = Any[]
	u_states = Any[]
	all_x = Any[]
	all_y = Any[]
	push!(xpositions,[s_0[1]])
	push!(ypositions,[s_0[2]])
	push!(u_states,u_0)
	s = deepcopy(s_0)
	u = deepcopy(u_0)
	for t in 1:max_t
		actions,n_actions = optimal_policy_q(s,u,opt_value,env)
		action = rand(actions)
		s_p = transition_s(s,action,env)
		u_p = transition_u(s,u,action,env)
		s = s_p
		u = u_p
		push!(xpositions,[s[1]])
		push!(ypositions,[s[2]])
		push!(all_x,s[1])
		push!(all_y,s[2])
		push!(u_states,u)		
	end
	xpositions,ypositions,u_states,all_x,all_y
end

# ╔═╡ 6e7b6b2a-5489-4860-930e-47b7df014840
md"## Animation"

# ╔═╡ 2ed5904d-03a3-4999-a949-415d0cf47328
md"Produce animation? $(@bind movie_q CheckBox(default = false))"

# ╔═╡ 5b1ba4f6-37a9-4e61-b6cc-3d495aa67c9d
md"# Comparison between agents"

# ╔═╡ bb45134a-88b1-40d2-a486-c7afe8ac744e
begin
	q_value_50 = readdlm("q_value_u_50.dat")
	h_value_50 = readdlm("h_value_u_50.dat")
	q_value_200 = readdlm("q_value_u_200.dat")
	h_value_200 = readdlm("h_value_u_200.dat")
	q_value_50 = reshape(q_value_50,env1.sizex,env1.sizey,env1.sizeu)
	h_value_50 = reshape(h_value_50,env1.sizex,env1.sizey,env1.sizeu)
	q_value_200 = reshape(q_value_200,env1.sizex,env1.sizey,env2.sizeu)
	h_value_200 = reshape(h_value_200,env1.sizex,env1.sizey,env2.sizeu)
end;

# ╔═╡ f1d5ee65-10f5-424a-b018-2aff1a5d7ff8
begin
		#Initial condition
		s_0 = [6,6] #[1,env.sizey]
		u_0 = 50# Int(env1.sizeu)
end

# ╔═╡ bd16a66c-9c2f-449c-a792-1073c54e990b
begin
	draw_environment([s_0[1]],[s_0[2]],[u_0],env2)
	savefig("arena.pdf")
end

# ╔═╡ a0729563-0b6d-4014-b8c7-9eb284a34606
if movie
	h_xpos_anim,h_ypos_anim,h_us_anim,h_allx_anim,h_ally_anim,h_urgency_anim = sample_trajectory(s_0,u_0,h_value,max_t_anim,env1)
	anim_h = animation(h_xpos_anim,h_ypos_anim,h_us_anim,max_t_anim,env1)
	gif(anim_h, "h_agent.gif", fps = 8)
end

# ╔═╡ b49b6396-c38a-462c-a9cb-177cb7c2b038
histogram(h_urgency_anim,bins = collect(1:env1.sizeu), leg = false, ylabel = "times p(action) > 0.6", xlabel = "u", normed = true)

# ╔═╡ 787bbe73-6052-41e0-bc8c-955e4a884886
if movie_q
	q_xpos_anim,q_ypos_anim,q_us_anim,q_allx_anim,q_ally_anim= sample_trajectory_q(s_0,u_0,q_value,max_t_anim,env1)
	anim_q = animation(q_xpos_anim,q_ypos_anim,q_us_anim,max_t_anim,env1)
	gif(anim_q, "q_agent.gif", fps = 8)
end

# ╔═╡ 693922e7-65f0-4350-9d5e-610ad0da7680
u_0

# ╔═╡ 78a5caf6-eced-4783-b950-26563f632be2
begin
	max_t = 100000
end;

# ╔═╡ 4a868ec2-b636-4d5d-a248-0a4e0cca3668
begin
	h_xpos_50,h_ypos_50,h_us_50,h_allx_50,h_ally_50,h_urgency_50 = sample_trajectory(s_0,u_0,h_value_50,max_t,env1)
	h_xpos_200,h_ypos_200,h_us_200,h_allx_200,h_ally_200,h_urgency_200 = sample_trajectory(s_0,u_0,h_value_200,max_t,env2)
end

# ╔═╡ 6edb5b48-b590-492d-bd3e-6a4f549aae30
begin
	q_xpos_50,q_ypos_50,q_us_50,q_allx_50,q_ally_50 = sample_trajectory_q(s_0,u_0,q_value_50,max_t,env1)
	q_xpos_200,q_ypos_200,q_us_200,q_allx_200,q_ally_200 = sample_trajectory_q(s_0,u_0,q_value_200,max_t,env2)
end;

# ╔═╡ 567f6b5d-c67e-4a43-9699-5625f1cc21a4
md"### Histogram of visited external states"

# ╔═╡ 9d9801e4-42a9-46b0-ba4e-253b276e5e21
theme(:vibrant)

# ╔═╡ 3e8a7dbb-8dfa-44f8-beab-ea32f3d478b4
begin
	p_h50_hist = plot(ticks = false, title = "H agent",ylabel = "\$u_{max} = 50\$")
	#plot!(p_hist1,env1.obstaclesx,env1.obstaclesy, st = :histogram2d)
	heatmap!(p_h50_hist,zeros(env1.sizex,env1.sizey))
	histogram2d!(p_h50_hist,h_allx_50,h_ally_50, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = :pdf,clim = (0,0.2),cbar = false)
	p_q50_hist = plot(ticks = false, title = "Q agent")
	heatmap!(p_q50_hist,zeros(env1.sizex,env1.sizey))
	histogram2d!(p_q50_hist, q_allx_50,q_ally_50, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = :true,clim = (0,0.2))
	
	p_h200_hist = plot(ticks = false)
	#plot!(p_hist1,env1.obstaclesx,env1.obstaclesy, st = :histogram2d)
	heatmap!(p_h200_hist,zeros(env1.sizex,env1.sizey))
	histogram2d!(p_h200_hist,h_allx_200,h_ally_200, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = true,clim = (0,0.2),cbar = false,ylabel = "\$u_{max} = 200\$")
	p_q200_hist = plot(ticks = false)
	heatmap!(p_q200_hist,zeros(env1.sizex,env1.sizey))
	histogram2d!(p_q200_hist, q_allx_200,q_ally_200, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = true,clim = (0,0.2))
	plot(p_h50_hist,p_q50_hist,p_h200_hist,p_q200_hist, layout = Plots.grid(2, 2, widths=[0.45,0.55]),size = (800,700),margin = 4Plots.mm)
	#savefig("histogram_space.png")
end

# ╔═╡ 1483bfe5-150d-40f5-b9dc-9488dcbc88b2
md"### Histogram of visitation of internal states"

# ╔═╡ d93ef435-c460-40cc-96e7-817e9eaace55
begin
	p_us = plot(xlim = (0,200), xlabel = "Internal energy \$u\$", ylabel = "Density",legend_foreground_color = nothing, margin = 5Plots.mm, majorgrid = false,size = (600,300))
	bd = 1
	plot!(p_us,q_us_200, label = "Q agent, \$u_{max} = 200\$", bandwidth = bd, st = :density,linewidth = 2)
	plot!(p_us,h_us_200, label = "H agent, \$u_{max} = 200\$", bandwidth = bd, st = :density,linewidth = 2)
	plot!(p_us,q_us_50, label = "Q agent, \$u_{max} = 50\$", bandwidth = bd, st = :density,linewidth = 3,ls = :dash)
	plot!(p_us,h_us_50, label = "H agent, \$u_{max} = 50\$", bandwidth = bd, st = :density,linewidth = 3,ls = :dash)
	#savefig("histogram_us.pdf")
end

# ╔═╡ Cell order:
# ╠═a649da15-0815-438d-9bef-02c6d204656e
# ╠═25c56490-3b9c-4825-b91d-8b9e41fc0f6b
# ╠═422493c5-8a90-4e70-bd06-40f8e6b254f1
# ╠═76f77726-7776-4975-9f30-3887f13ae3e7
# ╠═393eaf2d-e8fe-4675-a7e6-32d0fe9ac4e7
# ╠═b4e7b585-261c-4044-87cc-cbf669768145
# ╟─7feeec1a-7d7b-4220-917d-049f1e9b101b
# ╟─7e68e560-45d8-4429-8bff-3a8229c8c84e
# ╟─194e91cb-b619-4908-aebd-3136107175b7
# ╟─a46ced5b-2e58-40b2-8eb6-b4840043c055
# ╟─9404080e-a52c-42f7-9abd-ea488bf7abc2
# ╟─0dcdad0b-7acc-4fc4-93aa-f6eacc077cd3
# ╟─0ce119b1-e269-41e2-80b7-991cae37cf5f
# ╟─8675158f-97fb-4222-a32b-49ce4f6f1d41
# ╟─92bca36b-1dc9-4c03-88c0-6a684dfbec9f
# ╟─c96e3331-1dcd-4b9c-b28d-d74493c8934d
# ╟─d0a5c0fe-895f-42d8-9db6-3b0fcc6bb43e
# ╟─6c716ad4-23c4-46f8-ba77-340029fcce87
# ╠═07abd5b7-b465-425b-9823-19b73d07db56
# ╠═8f2fdc23-1b82-4479-afe7-8eaf3304a122
# ╠═fa33c170-e46e-444b-92de-cd80327a7b0a
# ╠═5e6964a3-7169-4c0d-b6d3-3f11d3cc7d36
# ╠═403a06a7-e30f-4aa4-ade1-55dee37cd514
# ╠═bd16a66c-9c2f-449c-a792-1073c54e990b
# ╟─ac3a4aa3-1edf-467e-9a47-9f6d6655cd04
# ╟─c6870051-0241-4cef-9e5b-bc876a3894fa
# ╟─d88e0e27-2354-43ad-9c26-cdc90beeea0f
# ╟─184636e2-c87d-4a89-b231-ff4aef8424d5
# ╠═82fbe5a0-34a5-44c7-bdcb-36d16f09ea7b
# ╠═e67db730-ca7c-4ef4-a2d2-7e001d5f7a79
# ╠═73722c01-adee-4bfd-97b4-60f2ced23725
# ╟─76f506dc-b21d-4e13-a8e8-9d1b3bd21b30
# ╟─aa5e5bf6-6504-4c01-bb36-df0d7306f9de
# ╟─ef9e78e2-d61f-4940-9e62-40c6d060353b
# ╟─a4457d71-27dc-4c93-81ff-f21b2dfed41d
# ╟─7ad00e90-3431-4e61-9a7f-efbc14d0724e
# ╟─b072360a-6646-4d6d-90ea-716085c53f66
# ╠═a0729563-0b6d-4014-b8c7-9eb284a34606
# ╟─f45ca37a-cba5-41e8-8058-1138e58daf73
# ╠═b49b6396-c38a-462c-a9cb-177cb7c2b038
# ╟─f98d6ea0-9d98-4940-907c-455397158f3b
# ╟─5f4234f5-fc0e-4cdd-93ea-99b6463b2ba1
# ╟─7a0173ac-240d-4f93-b413-45c6af0f4011
# ╠═caadeb3b-0938-4559-8122-348c960a6eb1
# ╠═29a4d235-8b03-4701-af89-cd289f212e7d
# ╠═819c1be2-339f-4c37-b8a3-9d8cb6be6496
# ╠═358bc5ca-c1f6-40f1-ba2d-7e8466531903
# ╟─40d62df0-53bb-4b46-91b7-78ffd621a519
# ╠═005720d3-5920-476b-9f96-39971f512452
# ╠═2379dcc3-53cb-4fb6-b1e8-c851e36acd1f
# ╟─6e7b6b2a-5489-4860-930e-47b7df014840
# ╟─2ed5904d-03a3-4999-a949-415d0cf47328
# ╠═787bbe73-6052-41e0-bc8c-955e4a884886
# ╠═693922e7-65f0-4350-9d5e-610ad0da7680
# ╟─5b1ba4f6-37a9-4e61-b6cc-3d495aa67c9d
# ╠═bb45134a-88b1-40d2-a486-c7afe8ac744e
# ╠═f1d5ee65-10f5-424a-b018-2aff1a5d7ff8
# ╠═78a5caf6-eced-4783-b950-26563f632be2
# ╠═4a868ec2-b636-4d5d-a248-0a4e0cca3668
# ╠═6edb5b48-b590-492d-bd3e-6a4f549aae30
# ╟─567f6b5d-c67e-4a43-9699-5625f1cc21a4
# ╠═9d9801e4-42a9-46b0-ba4e-253b276e5e21
# ╠═3e8a7dbb-8dfa-44f8-beab-ea32f3d478b4
# ╟─1483bfe5-150d-40f5-b9dc-9488dcbc88b2
# ╠═d93ef435-c460-40cc-96e7-817e9eaace55
