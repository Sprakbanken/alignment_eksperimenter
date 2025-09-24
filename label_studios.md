### Hvordan sette opp label-studio på egen maskin

Selve installasjonen er veldig enkel, og dokumentasjonen (som jeg fulgte) på å installere Label Studios finnes her: https://labelstud.io/guide/quick_start

1) ### Installasjon og innlogging
    pip install label-studio
    label-studio start

2) Etter å ha kjørt kommandoen label-studio start blir dere tatt med til startsiden og bedt om å opprette en bruker/logge inn

### Lage et nytt prosjekt
1) Inne i Label Studio trykker dere på Create Project øverst til venstre
2) Gi prosjektet et navn, og trykk Save

### Dataimport
1) Inne i Create Project er det en tab som heter Data import. Trykk på den og last opp filen dere vil bruke. 

### Labeling Setup
1) Inne i Create Project er det en annen tab som heter Labeling Setup. Når dere trykker på den kommer dere til en meny med masse valg. Velg custom template      nederst i menyen, og lim inn denne koden i vinuet som popper opp og trykk Save: 

<View>
  <Header value="Er disse tekstene parallelle?"/>

  <Choices name="aligned" toName="lang1" choice="single">
    <Choice value="True"/>
    <Choice value="False"/>
    <Choice value="Almost parallel"/>
    <Choice value="Something is wrong"/>
  </Choices>

  <Style>
    .parallel-container {
      display: flex;
      gap: 20px;
    }
    .parallel-block {
      flex: 1;
    }
  </Style>

  <View className="parallel-container">
    <View className="parallel-block">
      <Header value="LANG 1"/>
      <Text name="lang1" value="$fulltext_joined_lang_1"/>
    </View>
    <View className="parallel-block">
      <Header value="LANG 2"/>
      <Text name="lang2" value="$fulltext_joined_lang_2"/>
    </View>
  </View>
</View>


