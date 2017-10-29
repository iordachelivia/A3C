local game = require 'dmlab.system.game'
local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local random = require 'common.random'
local custom_observations = require 'decorators.custom_observations'
local timeout = require 'decorators.timeout'
local api = {}

function api:start(episode, seed)
  make_map.seedRng(seed)
  api._count = 0
  api._finish_count = 0
  api._has_goal = false
  api.hasEpisodeFinishedMaze = false
  api.wait_frame = 12
end

function api:commandLine(oldCommandLine)
  return make_map.commandLine(oldCommandLine)
end

function api:createPickup(className)
  return pickups.defaults[className]
end

function api:pickup(spawn_id)
  api._count = api._count + 1
  if not api._has_goal and api._count == api._finish_count then
    api.hasEpisodeFinishedMaze = true
    api.wait_frame = 1
  end
end

function api:updateSpawnVars(spawnVars)
  local classname = spawnVars.classname
  if spawnVars.random_items then
    local possibleClassNames = helpers.split(spawnVars.random_items, ',')
    if #possibleClassNames > 0 then
      classname = possibleClassNames[
        random.uniformInt(1,  #possibleClassNames)]
    end
  end
  local pickup = pickups.defaults[spawnVars.classname]
  if pickup then
    if pickup.type == pickups.type.kReward and pickup.quantity > 0 then
      api._finish_count = api._finish_count + 1
      spawnVars.id = tostring(api._finish_count)
    end
    if pickup.type == pickups.type.kGoal then
      api._has_goal = true
    end
  end

  return spawnVars
end

function api:nextMap()
    p_position = random.uniformInt(0,  4)
    if p_position == 0 then
        map = "******\n**P A*\n** ***\n** ***\n*A ***\n******"
    end
    if p_position == 1 then
        map = "******\n** PA*\n** ***\n** ***\n*A ***\n******"
    end
    if p_position == 2 then
        map = "******\n**  A*\n**P***\n** ***\n*A ***\n******"
    end
    if p_position == 3 then
        map = "******\n**  A*\n** ***\n**P***\n*A ***\n******"
    end
    if p_position == 4 then
        map = "******\n**  A*\n** ***\n** ***\n*AP***\n******"
    end
    return make_map.makeMap("small_maze", map)
end
episodeLengthSeconds=30
timeout.decorate(api, episodeLengthSeconds)

local hasEpisodeFinished = api.hasEpisodeFinished
function api:hasEpisodeFinished(time_seconds)
    timeRemaining = episodeLengthSeconds - time_seconds
    if api.hasEpisodeFinishedMaze then
      api.wait_frame = api.wait_frame - 1
    end
    if api.wait_frame < 0 then
      return true
    end
    return hasEpisodeFinished and hasEpisodeFinished(api, time_seconds) or
           timeRemaining <= 0
end

return api

