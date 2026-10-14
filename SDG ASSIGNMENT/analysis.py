import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.default_rng(42)
OUT = "/home/claude/work/"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,
                     "axes.grid":True,"grid.alpha":.25})
C = ["#1f5fa8","#d1495b","#2a9d8f","#e9a03b","#7b5ea7","#4d908e","#8d6e63","#555555","#6a994e","#c06c84"]

# ---------- 1. DATA COLLECTION (simulated panel, 10 sectors x 2010-2025) ----------
years = np.arange(2010, 2026)
gdp = np.array([8.5,5.2,5.5,6.4,7.4,8.0,8.3,6.8,6.5,3.9,-5.8,9.7,7.0,8.2,6.5,6.3])
sectors = {  # base employment (million), base growth %, automation sensitivity, skill-demand slope
 "IT & Software":(3.0, 6.0, 0.02, 1.00), "Healthcare":(4.5, 3.5, 0.02, 0.50),
 "Renewable Energy":(0.4, 8.0, 0.03, 0.80), "E-commerce & Logistics":(2.0, 5.5, 0.05, 0.60),
 "Education":(5.0, 2.0, 0.02, 0.30), "Financial Services":(3.5, 3.0, 0.08, 0.50),
 "Construction":(12.0, 2.5, 0.04, 0.10), "Manufacturing":(18.0, 0.8, 0.10, 0.20),
 "Agriculture":(45.0,-1.2, 0.05, 0.05), "Retail":(8.0, 1.5, 0.08, 0.15)}
rows=[]
for s,(e0,g0,asens,ss) in sectors.items():
    emp=e0; auto=10+rng.normal(0,1)
    skill=20+rng.normal(0,1); post=100
    for i,y in enumerate(years):
        auto += 3.2+rng.normal(0,.5)               # automation index rises every year
        skill += 2.0*(1+ss)+rng.normal(0,.6)      # digital-skill demand rises
        g = g0 + 0.25*(gdp[i]-6.5) + 0.35*ss*(skill-20)/10 - asens*(auto-10)/3 + rng.normal(0,.5)
        if i>0: emp *= (1+g/100)
        post = emp*(1+0.4*g/100)*100/e0 + rng.normal(0,1.5)
        rows.append([y,s,emp,gdp[i],auto,skill,post,3+0.3*g+rng.normal(0,.4)])
raw = pd.DataFrame(rows,columns=["year","sector","employment_m","gdp_growth","automation_index",
                                 "digital_skill_demand","job_postings_index","wage_growth"])
# inject realistic data-quality problems so preprocessing is genuine
raw.loc[rng.choice(raw.index,6,replace=False),"automation_index"]=np.nan
raw.loc[rng.choice(raw.index,4,replace=False),"digital_skill_demand"]=np.nan
raw = pd.concat([raw, raw.sample(3,random_state=1)], ignore_index=True)   # duplicates
raw.loc[[25,97],"employment_m"] *= 10                                       # entry errors
raw.to_csv(OUT+"employment_raw.csv",index=False)

# ---------- 2. PREPROCESSING ----------
q = {}
q["rows_raw"]=len(raw); q["missing_raw"]=int(raw.isna().sum().sum()); q["dups"]=int(raw.duplicated().sum())
df = raw.drop_duplicates().sort_values(["sector","year"]).reset_index(drop=True)
# outliers: modified z-score of employment ratio to sector median (vectorised)
med = df.groupby("sector")["employment_m"].transform("median")
ratio = df["employment_m"]/med
mad = np.median(np.abs(ratio-np.median(ratio)))
mz = 0.6745*(ratio-np.median(ratio))/mad
# robust rule: values >5x the sector median are impossible jumps
out = (df["employment_m"]/med) > 5
q["outliers"]=int(out.sum())
df.loc[out,"employment_m"]=np.nan
num=["employment_m","automation_index","digital_skill_demand"]
df[num]=df.groupby("sector")[num].transform(lambda s:s.interpolate(limit_direction="both"))
q["missing_clean"]=int(df.isna().sum().sum()); q["rows_clean"]=len(df)
df["sector"]=df["sector"].astype("category")
df.to_csv(OUT+"employment_clean.csv",index=False)

# ---------- 3. NUMPY / PANDAS ANALYSIS ----------
df["growth_pct"]=df.groupby("sector",observed=True)["employment_m"].pct_change()*100
g = df.groupby("sector",observed=True)
summ = g.agg(emp_2010=("employment_m","first"),emp_2025=("employment_m","last"),
             avg_growth=("growth_pct","mean"))
summ["cagr_pct"]=((summ.emp_2025/summ.emp_2010)**(1/15)-1)*100
summ["net_change_m"]=summ.emp_2025-summ.emp_2010
summ=summ.sort_values("cagr_pct",ascending=False)
summ.round(2).to_csv(OUT+"sector_summary.csv")
tot=df.groupby("year")["employment_m"].sum()
q["total_2010"]=float(tot.loc[2010]); q["total_2025"]=float(tot.loc[2025])
share=df.pivot(index="year",columns="sector",values="employment_m")
share_pct=share.div(share.sum(axis=1),axis=0)*100
corr=df[["growth_pct","gdp_growth","automation_index","digital_skill_demand","job_postings_index","wage_growth"]].corr()
q["corr_growth"]=corr["growth_pct"].round(2).to_dict()

# ---------- 4. MACHINE LEARNING ----------
# predict growth in year t from information known at year t-1
feat=["growth_pct","gdp_growth","automation_index","digital_skill_demand","job_postings_index"]
d=df.sort_values(["sector","year"]).copy()
for f in feat: d["lag_"+f]=d.groupby("sector",observed=True)[f].shift(1)
d["lag_post_chg"]=d.groupby("sector",observed=True)["job_postings_index"].pct_change().shift(1)*100
X_cols=["lag_"+f for f in feat]+["lag_post_chg"]
sec_d=pd.get_dummies(d["sector"],prefix="sec").astype(float); d=pd.concat([d,sec_d],axis=1); X_cols=X_cols+list(sec_d.columns)
d=d.dropna(subset=X_cols+["growth_pct"])
train=d[d.year<=2021]; test=d[d.year>=2022]
models={"Naive (last year's growth)":None,"Linear Regression":make_pipeline(StandardScaler(),Ridge(alpha=1.0)),
        "Random Forest":RandomForestRegressor(300,max_depth=5,random_state=0),
        "Gradient Boosting":GradientBoostingRegressor(n_estimators=200,max_depth=2,learning_rate=.05,random_state=0)}
res={}; preds={}
for n,m in models.items():
    if m is None: p=test["lag_growth_pct"].values
    else: m.fit(train[X_cols],train["growth_pct"]); p=m.predict(test[X_cols])
    preds[n]=p
    res[n]={"MAE":mean_absolute_error(test.growth_pct,p),"R2":r2_score(test.growth_pct,p)}
resdf=pd.DataFrame(res).T.round(3); resdf.to_csv(OUT+"model_results.csv")
best=resdf.drop(index="Naive (last year's growth)")["MAE"].idxmin(); q["best"]=best; q["n_train"]=len(train); q["n_test"]=len(test)
rf=models["Random Forest"]
imp=pd.Series(rf.feature_importances_,index=X_cols)
imp=imp[[c for c in X_cols if not c.startswith("sec_")]].sort_values()
q["top_features"]=imp.sort_values(ascending=False).head(3).round(3).to_dict()

# trend forecast 2026-2030 per sector (NumPy polyfit on log employment)
fy=np.arange(2026,2031); fc={}
for s,gp in df.groupby("sector",observed=True):
    coef=np.polyfit(gp.year-2010,np.log(gp.employment_m),1)
    fc[s]=np.exp(np.polyval(coef,fy-2010))
fcdf=pd.DataFrame(fc,index=fy)
fc_chg=((fcdf.loc[2030]/share.loc[2025])-1)*100
q["forecast_chg_2030"]=fc_chg.round(1).to_dict()
json.dump(q,open(OUT+"metrics.json","w"),indent=1,default=float)
print(resdf); print(summ.round(2)); print(json.dumps(q,indent=1,default=float))

# ---------- 5. CHARTS ----------
# Fig 1 pipeline diagram
fig,ax=plt.subplots(figsize=(9,2.6)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,3)
steps=["1 Collect\ndata","2 Clean &\nprepare","3 Feature\nengineering","4 Train ML\nmodels","5 Validate on\nfuture years","6 Forecast &\ninform policy"]
for i,t in enumerate(steps):
    x=0.1+i*1.65
    ax.add_patch(FancyBboxPatch((x,0.9),1.4,1.2,boxstyle="round,pad=0.05",fc=C[0] if i<5 else C[2],ec="none"))
    ax.text(x+.7,1.5,t,ha="center",va="center",color="white",fontsize=9,fontweight="bold")
    if i<5: ax.annotate("",xy=(x+1.62,1.5),xytext=(x+1.45,1.5),arrowprops=dict(arrowstyle="->",color="#444"))
ax.set_title("How machine learning predicts employment trends: the workflow",fontsize=11,loc="left")
fig.savefig(OUT+"charts/fig1_workflow.png",bbox_inches="tight"); plt.close()
# Fig 2 employment index by sector
fig,ax=plt.subplots(figsize=(8,4.5)); idx=share/share.loc[2010]*100
order=idx.loc[2025].sort_values(ascending=False).index
ypos=idx.loc[2025,order].values.astype(float).copy()
for k in range(1,len(ypos)):
    ypos[k]=min(ypos[k],ypos[k-1]-9)
for i,s in enumerate(order):
    ax.plot(idx.index,idx[s],color=C[i],lw=2); ax.text(2025.2,ypos[i],s,color=C[i],va="center",fontsize=8)
ax.axvline(2020,color="#999",ls="--",lw=1); ax.text(2020.1,ax.get_ylim()[0]+5,"2020 shock",fontsize=8,color="#666")
ax.set_xlim(2010,2029); ax.set_xticks(range(2010,2026,5)); ax.set_ylabel("Employment index (2010 = 100)"); ax.set_title("Employment has diverged sharply across sectors",loc="left")
fig.savefig(OUT+"charts/fig2_trends.png",bbox_inches="tight"); plt.close()
# Fig 3 CAGR bar
fig,ax=plt.subplots(figsize=(7,4)); s=summ.cagr_pct.sort_values()
ax.barh(s.index,s.values,color=[C[1] if v<0 else C[0] for v in s.values])
for i,v in enumerate(s.values): ax.text(v+(.1 if v>=0 else -.1),i,f"{v:.1f}%",va="center",ha="left" if v>=0 else "right",fontsize=8)
ax.set_xlabel("Compound annual growth in employment, 2010–2025 (%)"); ax.set_title("Which sectors are growing and which are shrinking",loc="left"); ax.grid(axis="y",alpha=0)
fig.savefig(OUT+"charts/fig3_cagr.png",bbox_inches="tight"); plt.close()
# Fig 4 correlation heatmap
fig,ax=plt.subplots(figsize=(6.2,5)); lab=["Emp. growth","GDP growth","Automation","Digital skill\ndemand","Job postings","Wage growth"]
im=ax.imshow(corr.values,cmap="RdBu_r",vmin=-1,vmax=1); ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(lab,rotation=40,ha="right",fontsize=8); ax.set_yticklabels(lab,fontsize=8); ax.grid(False)
for i in range(6):
    for j in range(6): ax.text(j,i,f"{corr.values[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if abs(corr.values[i,j])>.6 else "black")
fig.colorbar(im,shrink=.8); ax.set_title("What moves with employment growth?",loc="left")
fig.savefig(OUT+"charts/fig4_corr.png",bbox_inches="tight"); plt.close()
# Fig 5 model comparison + actual vs predicted
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4))
a1.bar(range(4),resdf["MAE"],color=[C[7],C[3],C[0],C[2]]); a1.set_xticks(range(4)); a1.set_xticklabels(["Naive","Linear\nReg.","Random\nForest","Gradient\nBoosting"],fontsize=8)
for i,v in enumerate(resdf["MAE"]): a1.text(i,v+.03,f"{v:.2f}",ha="center",fontsize=8)
a1.set_ylabel("Mean absolute error (percentage points)"); a1.set_title("Test error by model (lower is better)",loc="left",fontsize=10); a1.grid(axis="x",alpha=0)
bp=preds[best]; a2.scatter(test.growth_pct,bp,color=C[0],alpha=.8,s=25)
lim=[min(test.growth_pct.min(),bp.min())-1,max(test.growth_pct.max(),bp.max())+1]; a2.plot(lim,lim,color=C[1],ls="--",lw=1)
a2.set_xlabel("Actual growth (%)"); a2.set_ylabel("Predicted growth (%)"); a2.set_title(f"{best}: actual vs predicted, 2022–25",loc="left",fontsize=10)
fig.tight_layout(); fig.savefig(OUT+"charts/fig5_models.png",bbox_inches="tight"); plt.close()
# Fig 6 feature importance
fig,ax=plt.subplots(figsize=(7,3.8)); nm={"lag_growth_pct":"Last year's growth","lag_gdp_growth":"GDP growth","lag_automation_index":"Automation index",
 "lag_digital_skill_demand":"Digital-skill demand","lag_job_postings_index":"Job postings","lag_post_chg":"Change in postings"}
ax.barh([nm[i] for i in imp.index],imp.values,color=C[0]); ax.set_xlabel("Importance (Random Forest)"); ax.set_title("Which signals does the model rely on?",loc="left"); ax.grid(axis="y",alpha=0)
fig.savefig(OUT+"charts/fig6_importance.png",bbox_inches="tight"); plt.close()
# Fig 7 forecast
fig,ax=plt.subplots(figsize=(7,4)); s=fc_chg.sort_values()
ax.barh(s.index,s.values,color=[C[1] if v<0 else C[2] for v in s.values])
for i,v in enumerate(s.values): ax.text(v+(1 if v>=0 else -1),i,f"{v:+.0f}%",va="center",ha="left" if v>=0 else "right",fontsize=8)
ax.set_xlabel("Projected change in employment, 2025 to 2030 (%)"); ax.set_title("Forecast: where jobs are heading by 2030",loc="left"); ax.grid(axis="y",alpha=0)
fig.savefig(OUT+"charts/fig7_forecast.png",bbox_inches="tight"); plt.close()
