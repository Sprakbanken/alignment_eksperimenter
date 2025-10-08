# Label studio 

## Installasjon og innlogging
```
uv sync --extra annotation
uv run label-studio start
```

Etter å ha kjørt kommandoen label-studio start blir dere tatt med til startsiden og bedt om å opprette en bruker/logge inn. Gjør dette (kan bruke tullemail)

### Lage et prosjekt og legg inn data
1) Inne i Label Studio trykker dere på Create Project øverst til venstre
2) Gi prosjektet et navn, og trykk Save
3) Inne i Create Project er det en knapp som heter Import. Trykk på den.
4) Trykk på Upload Files og last opp filen dere vil bruke.
5) Velg list of tasks i radioknappene
6) Trykk på den blå Import-knappen

### Labeling Setup
1) Trykk på Settings øverst til høyre
2) Velg Labeling Interface i menyen til venstre
3) Lim inn følgende snutt og trykk Save
```html
<View>
  <Header value="Er disse tekstene parallelle?"/>

  <Choices name="aligned" toName="lang1" choice="single">
    <Choice value="Parallell"/>
    <Choice value="Not parallell"/>
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
      <Header value="$lang_lang_1"/>
      <Text name="lang1" value="$fulltext_joined_lang_1"/>
    </View>
    <View className="parallel-block">
      <Header value="$lang_lang_2"/>
      <Text name="lang2" value="$fulltext_joined_lang_2"/>
    </View>
  </View>
</View>
```


