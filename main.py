import open3d as o3d
import numpy as np
import crochet as cro

#constructional classes -> stores data
    #the individual stitches of the pattern
class node():
    def __init__(self, parent, pointcoords, parentcoords, name):

        self.coords = pointcoords
        self.parent = parent #set the parent
        self.level = parent.level + 1 #set the level based on parents level
        self.name = name

        #euler fun times!
        difference = parentcoords - pointcoords
        h = np.linalg.norm(difference[:2]) #works out (x^2 + y^2)^0.5 (called h)
        self.phi = np.arctan2(difference[1], difference[0]) #works out the xy angle (rad) -> we need this to compute the order in which the stiches should be stitched
        self.theta = np.arctan2(h, difference[2]) #works out the hz angle -> probably redundant!
    
    def id(self, i):
        self.i = i
    
    #the startpoint of the crochet pattern
class king(): 
    def __init__(self, coords):
        self.coords = coords
        self.level = 0
        self.name = "ANY"

    def id(self, i):
        self.i = i

#organisational functions
    #generates the pointcloud and does some processing to make the stitches evenly spaced
def noun(path, height, p=10_000): #uses the point cloud and the constructional classes to create a network

    #the "height" is the straight line height of the model, measured in stitches
    #p is the number of points that is initially on the mesh before it is voxled
    #should be larger than the surface area of the required shape (in stitches)

    mesh = o3d.io.read_triangle_mesh(path) #read mesh

    o = mesh.sample_points_uniformly(number_of_points=p*5) #sample p*5 points points
    o = mesh.sample_points_poisson_disk(number_of_points=p, pcl=o) #reduce this to p points to make a uniform point cloud

    oheight = o.get_axis_aligned_bounding_box().get_extent()[2] #get the height of the pointcloud

    o.scale(height / oheight, center=(0,0,0)) #scale so 1 unit == 1 stitch

    o = o.voxel_down_sample(voxel_size=1) #voxelise to reduce stitch density to 1 stitch per stitch

    c = (np.array([o.colors]) * 255).astype("uint8") #make in the correct colour format
    cloud, npallet, names = cro.generatecloud(c, 5, 10)

    o.colors = o3d.utility.Vector3dVector(cloud)

    #visualise!
    o3d.visualization.draw_geometries([o])

    return o, npallet, names
def adjective(o, pallet, names): #uses the pointcloud to create the network
    
    a = np.asarray(o.points) #create variable of the points as a np array
    c = np.array(o.colors)
    h = a.T[2] #heights
    tree = o3d.geometry.KDTreeFlann(o) #make a pretty little tree (- V -)

    i = np.argmax(h) #find the highest point
    k = king(a[i]) #make a king of that point

    network = {
        "king": k, #for ease of acsess -> probs redundant
        "nodes": [k],
        "levels": 0
    }

    level = 0 #the current level we are working on
    while True:

        nodes = [n for n in network["nodes"] if n.level == level] #get the nodes from the network that are the working level
        if len(nodes) == 0:
            network["levels"] = level
            break #break if we are already done
        ncoords = [n.coords for n in nodes] #get the coords of the nodes (this is for later)
        allcoords = [tuple(n.coords) for n in network["nodes"]]
        selection = []

        for n in nodes:
            height = tuple(n.coords)[2] #get the nodes z height
            selected = list(tree.search_radius_vector_3d(n.coords, 1.2)[1]) #find all points within n stitches (adjustable value)
            for p in selected: #look through found points
                coords = tuple(a[p]) #get the point coords as a tuple
                if h[p] <= height and coords not in allcoords and p not in selection: #if lower than node, not already a node and not already selected...
                    selection.append(p) #select the point

        for point in selection:
            coords = a[point] #get the coords of the point
            hexa = "#"+("".join([hex(round(x * 255))[2:] for x in c[point]]))
            print(hexa)
            colour = "<span style=\"colour:{hexa}\">" + names[np.argmin(np.linalg.norm(pallet-c[point], axis=1))] + "</span>"

            i = np.argmin(np.linalg.norm(ncoords-coords, axis=1)) #get the index of the node with the minimum distance to the point
            parent = nodes[i] #get the parent
            network["nodes"].append(node(parent, coords, parent.coords, colour)) #make the node and put it in the network
            
        level += 1
        #print(level, len(nodes), len(network["nodes"]))

    return network
def verb(network): #uses the network to create a line-by-line file

    network["nodes"].sort(key=lambda n: n.level) #sort the nodes by their level in the network
    for i in range(len(network["nodes"])): #id all the nodes -> could probably do this earlier in the programme
        network["nodes"][i].id(i)

    levels = [[] for _ in range(network["levels"])] #create a list to hold the nodes in each level
    for n in network["nodes"]:
        levels[n.level].append(n) #put the nodes in their corrisponding index of "levels"

    with open("pattern.txt", "w") as f:
        for i in range(len(levels)-1): #itterate over all the levels of the pattern (except the first)
            
            level = levels[i+1] #list of the current level
            lower = levels[i] #list of the lower level (parents)

            ordered = [] #create the list that wil become levels[i+1]
            stitches = [] #number of stitches for each parent
            colours = []
            for node in lower:
                c = [n for n in level if n.parent == node] #get the list of children in the level that have the parent node
                #above possible weak point
                colours.append(node.name)
                stitches.append(len(c))#append to the number of stitches the number of childrent the parent node has
                c.sort(key=lambda n: n.phi) #sort by the angle to the parent
                    #I dont think it matters whether this list is reversed or not
                    #It will sort anti-clockwise but it also determines the order in which nodes are read so the two cancel out
                    #Idk tho this is all year-old spagetti code to me
                ordered += c

                #this block above was the problem code before:
                #by sorting by angle to parent across all nodes it basically fucked up anything that isnt a circle
                #instead what needed to happen was sort by angle for each of the parents, becasue we start with a single node
                #we can just run the sorter on each level then assume the parents to be in order when we do the next level

            if i % 2 != 0: stitches.reverse() #reverse the stitch list if were on an odd level

            for s in range(len(stitches)): #run through stitches
                if stitches[s] == 0: #if theres none
                    u = s+1 if s+1 < len(stitches) else 0 #set the next stitch index (loopback if at the end of the list)
                    d = s-1 #set the last stitch index
                    if max(stitches[d], stitches[u]) > 1: #find out if at least one of the stitches is larger than one
                        if stitches[d] >= stitches[u]: #if its the last stitch
                            stitches[d] -= 1 #decrement the last stitch
                        else: #if its the next stitch
                            stitches[u] -= 1 #decrement the next stitch
                        stitches[s] = 1 #increment the current stitch (its already 0 so it has to be 1)

            f.write(f"EOL\n") #write the starting line
            for i in range(len(stitches)): #run through the stitches
                c = stitches[i]
                f.write(f"Stitch {c} {colours[i]} on loop {i+1}\n")

if __name__ == "__main__":

    path = "Tawana.ply"
    height = 50 #the straight line height of the model in stitches

    o, pallet, names = noun(path, height) #get a pointmap with a heigh
    n = adjective(o, pallet, names) #passes adjective the points from noun
    verb(n) #make the line-by-line
