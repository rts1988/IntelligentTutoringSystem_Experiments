import streamlit as st

# generates a checklist of possible objectives useing the goal concepts. Arranged in descedning order of difficulty.
# when checked, the list of objectives is saved, and an explanation along with learning strategies is displayed under the checkbox, aligned with tab.
# study plan is generated based on list of objectives, and then a radius of important concepts to the goal is generated, and imporatnce of pathway concepts is scored based on how importantt to most of the goal concepts,
# and the role of the parthway concepts, also taking into account clt, and known_concepts.

objectivesdict = dict()
# objectivesdict['generating'] = dict()
# objectivesdict['generating']['phen'] = dict()
#
# objectivesdict['generating']['phen'][0] = ['Learn to generate ideas to use, control or respond to the phenomenon of ','conceptx']
# objectivesdict['generating']['phen'][1] = ['Learn to generate hypotheses to predict/explain the mechanism or behavior of phenomenon of ','conceptx', ', if it is not fully understood']
#
objectivesdict['recalling'] = dict()
objectivesdict['recalling']['phen'] = dict()
objectivesdict['recalling']['phen'][0] = ['Learn what set of systems the phenomenon of ','conceptx', ' may occur in ']
objectivesdict['recalling']['phen'][1] = ['Learn what the causes (necessary and sufficient conditions) are for the phenomenon of ','conceptx',' to occur']
objectivesdict['recalling']['phen'][2] = ['Learn what the necessary and sufficient conditions are for the phenomenon of ','conceptx',' to stop']
objectivesdict['recalling']['phen'][3] = ['Learn what the initial and final states (effects) of a system are when the phenomenon of ','conceptx',' happens']
objectivesdict['recalling']['phen'][4] = ['Learn what the rules governing the phenomenon of ','conceptx',' are']

objectivesdict['recalling']['att'] = dict()
objectivesdict['recalling']['att'][0] = ['Learn what sets and systems ','conceptx',' provides information on']
objectivesdict['recalling']['att'][1] = ['Learn the symbol(s) usually used to denote ','conceptx']
objectivesdict['recalling']['att'][2] = ['Learn which sets or systems ','conceptx',' is used to define. (essential property)']
objectivesdict['recalling']['att'][3] = ['Learn what types of values ','conceptx', ' can take']
objectivesdict['recalling']['att'][4] = ['Learn what units of measurement are used with ','conceptx']
objectivesdict['recalling']['att'][5] = ['Learn what the possible values of ','conceptx',' are']
objectivesdict['recalling']['att'][6] = ['Learn what the range of values of ','conceptx',' is for different sets']
objectivesdict['recalling']['att'][7] = ['Learn whether the value of ','conceptx',' is constant or changeable']
objectivesdict['recalling']['att'][8] = ['Learn what tools are used to determine the value of ','conceptx']
objectivesdict['recalling']['att'][9] = ['Learn what the procedure is used to determine the value of ','conceptx']
objectivesdict['recalling']['att'][10] = ['Learn the rules/relationship between ','conceptx',' and other attributes in a system']
objectivesdict['recalling']['att'][11] = ['Learn how the history of a system impacts the value of ','conceptx']

objectivesdict['recalling']['ins'] = dict()
objectivesdict['recalling']['ins'][0] = ['Learn to be able to identify ','conceptx',' based on unique characteristics']
objectivesdict['recalling']['ins'][1] = ['Learn what other names/terms are used for ','conceptx']
objectivesdict['recalling']['ins'][2] = ['Learn the timing, location and event associated with origin of ','conceptx']
objectivesdict['recalling']['ins'][3] = ['Learn the timing, location and event associated with end/death of ','conceptx']
objectivesdict['recalling']['ins'][4] = ['Learn to describe the state of ','conceptx',' using its features']


objectivesdict['recalling']['set'] = dict()
objectivesdict['recalling']['set'][0] = ['Learn distinguishing features of members of the set ','conceptx']
objectivesdict['recalling']['set'][1] = ['Learn what other names/terms are used for ','conceptx']
objectivesdict['recalling']['set'][2] = ['Learn the meaning and definition of ','conceptx']
objectivesdict['recalling']['set'][3] = ['Learn to list members of the set of ','conceptx']
objectivesdict['recalling']['set'][4] = ['Learn to describe ','conceptx',' using its features, and the range of possible values of its attributes']

goal_concepts = ['anxiety','adhd','depression','individual','efficacy','Kolk','love']

isphen = ['anxiety','adhd','depression','love']
isset = ['anxiety','adhd','depression','individual']
isatt = ['efficacy']
isins = ['Kolk']

objectivescheck = dict()

for verb in objectivesdict:
    objectivescheck[verb] = dict()
    st.header(verb)
    for gc in goal_concepts:
        st.subheader(gc)
        objectivescheck[verb][gc] = dict()

        if gc in isatt:
             # consider having roles in a superset dict of Agg_phen_to_sent etc. and iterting among the keys of the superset dict here.
             objectivescheck[verb][gc]['att'] = dict()
             for (key,val) in objectivesdict[verb]['att'].items():
                 objectivescheck[verb][gc]['att'][key] = False
                 labelstring = ''
                 for s in val:
                     if s=='conceptx':
                         labelstring = labelstring+gc # here, replace with mcform
                     else:
                         labelstring = labelstring + s
                 objectivescheck[verb][gc]['att'][key] = st.checkbox(labelstring)


        if gc in isset:
             # consider having roles in a superset dict of Agg_phen_to_sent etc. and iterting among the keys of the superset dict here.
             objectivescheck[verb][gc]['set'] = dict()
             for (key,val) in objectivesdict[verb]['set'].items():
                 objectivescheck[verb][gc]['set'][key] = False
                 labelstring = ''
                 for s in val:
                     if s=='conceptx':
                         labelstring = labelstring+gc # here, replace with mcform
                     else:
                         labelstring = labelstring + s
                 objectivescheck[verb][gc]['set'][key] = st.checkbox(labelstring)



        if gc in isins:
             # consider having roles in a superset dict of Agg_phen_to_sent etc. and iterting among the keys of the superset dict here.
             objectivescheck[verb][gc]['ins'] = dict()
             for (key,val) in objectivesdict[verb]['ins'].items():
                 objectivescheck[verb][gc]['ins'][key] = False
                 labelstring = ''
                 for s in val:
                     if s=='conceptx':
                         labelstring = labelstring+gc # here, replace with mcform
                     else:
                         labelstring = labelstring + s
                 objectivescheck[verb][gc]['ins'][key] = st.checkbox(labelstring)



        if gc in isphen:
             # consider having roles in a superset dict of Agg_phen_to_sent etc. and iterting among the keys of the superset dict here.
             objectivescheck[verb][gc]['phen'] = dict()
             for (key,val) in objectivesdict[verb]['phen'].items():
                 objectivescheck[verb][gc]['phen'][key] = False
                 labelstring = ''
                 for s in val:
                     if s=='conceptx':
                         labelstring = labelstring+gc # here, replace with mcform
                     else:
                         labelstring = labelstring + s
                 objectivescheck[verb][gc]['phen'][key] = st.checkbox(labelstring)


objectivescheck
